#!/usr/bin/env python3

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
        self.exec_network = None
        self.infer_request = None
        self.supported_layers = None
        self.unsupported_layers = None

    def load_model(self, model, device="CPU", labels="None"):
        '''
        Load the model given IR files.
        Synchronous requests made within.
        '''

        # ------------- 1. Plugin initialization for specified device and load extensions library if specified -------------
        # Initialize the plugin
        # log.info("Creating Inference Engine...")
        self.plugin = IECore()

        # -------------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) --------------------
        # Read the IR as a IENetwork
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        # log.info("Loading network files:\n\t{}\n\t{}...".format(model_xml, model_bin))
        self.network = IENetwork(model=model_xml, weights=model_bin)

        # ------------3. Display Device Info, check for unsupported layer and add extesnion library if required ------------
        # Display Device info
        # log.info("Device info:")
        # versions = self.plugin.get_versions(device)
        # print("{}{}".format(" "*8, device))
        # print("{}MKLDNNPlugin version ......... {}.{}".format(" "*8, versions[device].major, versions[device].minor))
        # print("{}Build ........... {}".format(" "*8, versions[device].build_number))

        # Get the supported layers of the network
        # Check for any unsupported layers, and let the user
        # know if anything is missing. Exit the program, if so.
        if "CPU" in device:
            # log.info("Check for any unsupported layers...")
            self.supported_layers = self.plugin.query_network(network=self.network, device_name="CPU")
            self.unsupported_layers = [l for l in self.network.layers.keys() if l not in self.supported_layers]
            # if len(self.unsupported_layers) != 0:
            #     log.error("Following layers are not supported by the plugin for specified device:\n {}".
            #               format(not_supported_layers))
            #     log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
            #               "or --cpu_extension command line argument")
            #     sys.exit(1)

        # # Add a CPU extension, if applicable
        # if cpu_extension and "CPU" in device:
        #     self.plugin.add_extension(cpu_extension, device)

        # ----------------------------------------- 4. Loading model to the plugin -----------------------------------------
        # Load the IENetwork into the plugin
        # log.info("Loading model to the plugin...")
        self.exec_network = self.plugin.load_network(network=self.network, num_requests=2, device_name=device)

        # ---------------------------------------------- 5. Preparing inputs -----------------------------------------------
        # log.info("Preparing input and output blobs...")
        assert len(self.network.outputs) == 1, "Demo supports only single output topologies"
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        #  Defaulf batch_size is 1
        self.network.batch_size = 1

        return

    def get_input_shape(self):
        '''
        Gets the input shape of the network
        typically: n, c, h, w
        '''
        return self.network.inputs[self.input_blob].shape

    def async_inference(self, request_id, image):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        self.exec_network.start_async(request_id=request_id, 
            inputs={self.input_blob: image})
        return

    def wait(self, request_id):
        '''
        Checks the status of the inference request.
        '''
        status = self.exec_network.requests[request_id].wait(-1)
        return status

    def extract_output(self, request_id):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        return self.exec_network.requests[request_id].outputs[self.output_blob]
