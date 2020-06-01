"""People Counter."""
"""
###COMMAND###
 python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m person-detection-retail-0013/FP32/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image, video file. To use Videocamera write VIDEOCAMERA. ")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### Connect to the MQTT client ###
    client = None
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def handleInputStream(input):
    
    single_image_mode = False
    input_stream = input
    
    # Videocamera
    if input == 'VIDEOCAMERA':
        input_stream = 0

    # Image - select image mode
    elif input.endswith('.jpg') or input.endswith('.bmp') :
        single_image_mode = True

    # Video file - check if exists
    else:
        assert os.path.isfile(input), "[ERROR] Input file is invalid or unsupported"
    
    return input_stream,single_image_mode

def preprocess_frame(frame,n,c,h,w): 
    # Resize and change channels 
    image = cv2.resize(frame, (w, h))
    image = image.transpose((2, 0, 1))
    image = image.reshape((n, c, h, w))
    return image

def handle_output(frame, result, init_w,init_h,prob_threshold):
    #Draws output boxes from the result
    current_count = 0
    for obj in result[0][0]:
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * init_w)
            ymin = int(obj[4] * init_h)
            xmax = int(obj[5] * init_w)
            ymax = int(obj[6] * init_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
            current_count = current_count + 1
    return frame, current_count

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    ### INITIALIZE VARIABLES ###
    
    #Current request id 
    current_req=0
    
    #Current count of people
    current_count=0
    
    #Last count of people
    last_count=0
    
    #Total count of people
    total_count=0

    #Start time of person in fame
    start_time=0

    
    #Setting threshold
    prob_threshold = args.prob_threshold

    #Amount of frames it is going to wait  
    #before detecting a new person
    tol_threshold = 10

    #Flags for detecting a new person
    new_person = True
    tolerance =0
   
    
    ### INFERENCE ###
    
    # Initialise the class
    infer_network = Network()
   
    ### Load the model through `infer_network` ###
    n, c, h, w = infer_network.load_model(args.model, args.device,current_req, args.cpu_extension)[1]

    ### Handle the input stream ###
    
    input_stream, single_image_mode = handleInputStream(args.input)
      
    cap = cv2.VideoCapture(input_stream)
    
    if input_stream:
        cap.open(args.input)

    if not cap.isOpened():
        log.error("[ERROR] Missing video source")
        
    ###Setting initial width and height###
    init_w = cap.get(3)
    init_h = cap.get(4)

    ### Loop until stream is over ###
    while cap.isOpened():
                    
        ### Read from the video capture ###
        flag, frame = cap.read()
        
        ###Check if I need to exit the loop###
        if not flag:
            break
        key_pressed = cv2.waitKey(60)  
        

        ### Pre-process the image ###
        processed_frame = preprocess_frame(frame,n,c,h,w)
           
        ### Start asynchronous inference for specified request ###
        infer_network.exec_net(current_req, processed_frame)

        inf_start_time = time.time()


        ### Wait for the result ###
        if infer_network.wait(current_req) == 0:
            
            ##Calculate inference time 
            det_time = time.time() - inf_start_time
            ### Get result of the inference request ###
            result = infer_network.get_output(current_req)

            ###  Extract and draw boxes from the results ###
            frame, current_count = handle_output(frame, result, init_w, init_h,prob_threshold)

            ##Put the inference time in the frame ###
            time_message = "Inference time: {:.3f}ms".format(det_time * 1000)  
            cv2.putText(frame,time_message, (15, 15),cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

            ### Calculate and send relevant information to the MQTT server (current_count, total_count,duration)###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###   

                
            
            #Person enters the frame 
            if (current_count > last_count) and new_person:
                #Resets time
                start_time = time.time()
                #Sets the flag to calculate tolerance
                new_person = False
                #Recalculate total count
                total_count += (current_count-last_count)
                
                #Publish message to the MQTT server
                client.publish("person", json.dumps({"total": total_count}))
            
            #Person goes out the frame 
            if (current_count <= last_count) and not new_person:

                #Check some frames to see if the person has left:
                tolerance +=1

                if tolerance > tol_threshold:
                    #Calculate duration
                    duration = int(time.time() - start_time)

                    #Reset flags
                    new_person=True
                    tolerance=0

                    # Publish message to the MQTT server
                    client.publish("person/duration", json.dumps({"duration": duration}))

            client.publish("person", json.dumps({"count": current_count}))    
            last_count = current_count
            
            
            if key_pressed == 27:
                break  
            
        ### Send the frame to the FFMPEG server (assuming FFMPEG server is reading stdout)###
        
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()

        ### Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)
                    
            
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
