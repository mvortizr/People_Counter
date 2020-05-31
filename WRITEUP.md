
# Write Up


## Additional Information 

- The project was done and tested in a computer with these specs:
	- Ubuntu 18.04.4 LTS
	- RAM 3.7GB
	- 64bit
	- Intel Core i-5


## Custom layers

When we are converting a model into Intermediate Representation, it could happen that a layer of the model is unsupported by the Model Optimizer (MO) thankfully  OpenVino provide us a way to deal with it: Custom Layers.
Custom layers are an additional configuration that the OpenVino toolkit provides: If a layer of a model is not part of the [list of supported layers](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html),  we need to handle manually those unsupported layers instead of just relying on the MO.

__How is it done?__
It depends of the framework of the model. For TensorFlow and Caffe models, the unsupported layers can be registered as extensions of the MO. For TensorFlow we have the additional option of replacing the unsupported subgraph with a different subgraph. For Caffe models, another option is to register the layers as Custom, then use Caffe to calculate the output shape of the layer. 

I tried several models in a CPU and none of them required an additional configuration in their layers. It also helps that I added the CPU extension before checking for custom layers. 

__Reasons to handle custom layers__
Some of the potential reasons for handling custom layers are:
- A person wants to use some cutting edge technology that is still not supported for that particular device.
- A researcher wants to try their experiments on edge and see if OpenVino can provide a good result. 
- For any other reason, a person wants to deploy a project using OpenVino and the model has an unsupported layer (or several). 


### Compare Model Performance
Deploying the model using OpenVino improved the performance, I tracked two metrics: Model Size and Inference Time. 
 
__Model Size__

| Model         | Without OpenVino| With OpenVino  |
| -------------|:--------------:|----------------:|
| SSD MobileNet V2 COCO | 208.3 MB| 69.8MB |

__Inference Time__
| Model         | Without OpenVino| With OpenVino  |
| -------------|:--------------:|----------------:|
| SSD MobileNet V2 COCO |  150 ms| 127-140 ms |

### Model Use Cases
A model that detects humans on edge can be uselful in many scenarios, some of them are:
- Security System: Check how many people enter a space and trigger an alarm every time someone not authorized wants to enter. 
- Queue Management: Check how many people are in queue and assign numbers automatically, or make statistics about the amount of time they spent in queue and compare performance between different workers. 
- Lockdown detector: Check how many people are in some streets and not quarantined and how much they last outside. Maybe some trick around the bounding boxes could even detect if they are respecting the 6feet distance around each other.


### Effects on the End User
Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. 

- __Lighting__: Most models failed to detect people on dark spaces, this is because in the dark the pixels of each people and the scene are similar.  Also, the most popular image datasets don't have many night / low light photos so the models don't get many training at detecting people on these conditions.

- __Model accuracy__ : The hardware used to deploy models on the edge generally is not very powerful at processing floating points or images. So, developers and users have to find a proper accuracy that makes the model useful and at the same time doesn't put additional burdens on the hardware.

- __Camera focal length/image__ : Most models are trained using photos in high resolution or optimal conditions so could fail with a bad camera setup, it is important that the deployment environment can provide proper camera settings. Another option is to do transfer training with distorted images from pretrained model.

### Model Research

I found a suitable model but I still wanted to write this part.

I tried several models Object Detection models that used Bounding Boxes, the final project works with any SSD model that has an input shape [BxHxWxC] and outputs a box coordinates with their confidence.

1. __The First Model__ I tried was an SSD ONNX model [(Link)](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/ssd)
but I have problems to run the ONNX backend to make the inference comparison, so I decided to switch to TensorFlow Model Zoo

This model was converted using this command:
```python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py  --input_model ssd.onnx```

2. __The second model__ I tried was the Faster R-CNN Inception V2 COCO from the TensorFlow Model Zoo [(Link)](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz). I handled the inputs and outputs (bad-inference.py and bad-main.py files have that implementation) but my computer was extremely slow with it at a point in which I couldn't move the cursor.  As I don't have a good internet connection, it was really difficult to use the Udacity workspace.

- This model was converted using this command:
```python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json```

3. __My third option__ was the SSD MobileNet V2 COCO [(Link)](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz). __This was the chosen model__ as it ran smoothly on my computer and I was able to handle the inputs and outputs easily.

- This model was converted using this command:
```python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json```

Other models that I tested and could be used with this project:
 - SSD Lite MobileNet V2 COCO [(Link)](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz)
- person-detection-action-recognition-0006 (is already an IR) [(Link)](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_action_recognition_0006_description_person_detection_action_recognition_0006.html)

### Command used to run the app
```python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm```

This command assumes that the model located in a `/models/ssd_mobilenet_v2_coco_2018_03_29` folder and already converted in an Intermediate Representation
