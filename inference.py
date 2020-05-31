#!/usr/bin/env python3

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork,IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.infer_request = None
        self.exec_network = None

    def load_model(self, model, device, num_requests, cpu_extension=None, plugin=None):

        # Getting the reference of the model
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)

        # Initialize the plugin for the device
        if not plugin:
            self.plugin = IECore()
        else:
            self.plugin = plugin

        #Add CPU extension if applicable
        if cpu_extension and 'CPU' in device:
            self.plugin.add_extension(cpu_extension,device)

        #Check for unsupported layers 
        if device == "CPU":     
            supported_layers = self.plugin.query_network(network=self.network, device_name=device)  
            unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
            if len(unsupported_layers) != 0:
                log.error("[ERROR] Unsupported layers found: {}".format(unsupported_layers))
                sys.exit(1)

        # Loads network read from IR to the plugin   
        if num_requests == 0:          
            self.exec_network = self.plugin.load_network(self.network,device)
        else:
            self.exec_network = self.plugin.load_network(self.network, device, num_requests=num_requests)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return self.plugin, self.get_input_shape()
       

    def get_input_shape(self):
        ### Return the shape of the input layer ###
        return self.network.inputs[self.input_blob].shape

    def exec_net(self,request_id,frame):
        ### Start an asynchronous request ###
        self.infer_request= self.exec_network.start_async(request_id=request_id, inputs={self.input_blob: frame})
        return self.exec_network

    def wait(self,request_id):
        ###  Wait for the request to be complete. ###
        wait_process = self.exec_network.requests[request_id].wait(-1)
        return wait_process

    def get_output(self, request_id, output=None):
        ### Extract and return the output results
        if output:
            res = self.infer_request.outputs[output]
        else:
            res = self.exec_network.requests[request_id].outputs[self.output_blob]
        return res