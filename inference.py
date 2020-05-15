#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


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
        self.net_plugin = None

    def load_model(self, model, plugin=None, device="CPU",cpu_extension=None,num_requests):
        ### TODO: Load the model ###
        ### TODO: Check for supported layers ###
        ### TODO: Add any necessary extensions ###
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###


        # Getting the reference of the model
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin for the device
        if not plugin:
            self.plugin = IEPlugin(device = device)
        else:
            self.plugin = plugin


        #Add CPU extension if applicable
        if cpu_extension and 'CPU' in device:
            self.plugin.add_cpu_extension(cpu_extension)

       
        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)

        #Check for unsupported layers
        if self.plugin.device == "CPU":
            supported_layers = self.plugin.get_supported_layers(self.network)
                not_supported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                if len(not_supported_layers) != 0:
                    log.error("[ERROR] Some layers are not supported by CPU {}". format(' '.join(not_supported_layers)))
                    sys.exit(1)


        if num_requests == 0:
            # Loads network read from IR to the plugin
            self.net_plugin = self.plugin.load(network=self.network)
        else:
            self.net_plugin = self.plugin.load(network=self.network, num_requests=num_requests)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return self.plugin
       

    def get_input_shape(self):
        ### Return the shape of the input layer ###
        return self.network.inputs[self.input_blob].shape

    def exec_net(self,request_id,frame):
        ### Start an asynchronous request ###
        self.infer_request_handle = self.net_plugin.start_async(request_id=request_id, inputs={self.input_blob: frame})
        return self.net_plugin

    def wait(self,request_id):
        ###  Wait for the request to be complete. ###
        wait_process = self.net_plugin.requests[request_id].wait(-1)
        return wait_process

    def get_output(self, request_id, output=None):
        ### Extract and return the output results
        if output:
            res = self.infer_request_handle.outputs[output]
        else:
            res = self.net_plugin.requests[request_id].outputs[self.out_blob]
        return res
