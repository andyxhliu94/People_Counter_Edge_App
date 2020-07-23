"""People Counter App."""
from __future__ import print_function, division
import argparse
import os
import sys
import socket
import json
import cv2

import numpy as np
import logging as log
import paho.mqtt.client as mqtt

# from argparse import ArgumentParser, SUPPRESS
# import argparse
from math import exp as exp
from time import time
from inference_ssd import Network
from argparse import SUPPRESS
from random import randint

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

def get_args():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument_group('Options')
    parser.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    parser.add_argument('-m', "--model", required=False, type=str,
                        help="Required. Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=False, type=str,
                        help="Required. Path to image or video file or 'CAM' for camera")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="Optional. Required for CPU custom layers. MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Optional. Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-la", "--labels", help="Optional. Labels mapping file", default=None, type=str)
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Optional. Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("--no_show", help="Optional. Don't show output", action='store_true')
    args = parser.parse_args()

    return args


def draw_masks(result, width, height):
    '''
    Draw semantic mask classes onto the frame.
    '''
    # Create a mask with color by class
    classes = cv2.resize(result[0].transpose((1,2,0)), (width,height), 
        interpolation=cv2.INTER_NEAREST)
    unique_classes = np.unique(classes)
    out_mask = classes * (255/20)
    
    # Stack the mask so FFmpeg understands it
    out_mask = np.dstack((out_mask, out_mask, out_mask))
    out_mask = np.uint8(out_mask)

    return out_mask, unique_classes


def get_class_names(class_nums):
    class_names= []
    for i in class_nums:
        class_names.append(CLASSES[int(i)])
    return class_names

def connect_mqtt():
    # Connect to the MQTT server
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def infer_on_stream(args, client):

    ### Initialize the Inference Engine
    infer_network = Network()

    ### Load the network model into the IE
    infer_network.load_model(args.model, args.device, args.labels)
    n, c, h, w = infer_network.get_input_shape()

    ### Handle the input stream ###
    input_stream = 0 if args.input == "CAM" else args.input
    cap = cv2.VideoCapture(input_stream)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    #Initialize async variables
    cur_request_id = 0
    next_request_id = 1
    is_async_mode = True
    render_time = 0

    #In async mode, first read the current frame
    if is_async_mode:
        flag, frame = cap.read()
        frame_h, frame_w = frame.shape[:2]

    # Initialize useful stats params
    current_count = 0 #目前人数: Current people count
    last_count = 0 #之前人数: Last people count
    total_count = 0 #总人数: Total people count 
    average_duration = 0 #平均逗留时间: Average stay duration
    lagtime = 0 #没有正确检测到目标的延误时间：Lagtime due to not correctly detecting objects

    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        if is_async_mode:
            flag, next_frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        # Main sync point:
        # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
        # in the regular mode we start the CURRENT request and immediately wait for it's completion
        inf_start = time()
        # Pre-process the frame
        if is_async_mode:
            in_frame = cv2.resize(next_frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            # Perform inference on the frame
            infer_network.async_inference(request_id=next_request_id, image=in_frame)

        # Get the output of inference
        if infer_network.wait(request_id=cur_request_id) == 0:
            inf_end = time()
            det_time = inf_end - inf_start

             # Parse detection results of the current request
            result = infer_network.extract_output(cur_request_id)
            # Collecting object detection results
            objects = list()
            objects = [obj for obj in result[0][0] if obj[1] == 1] #only place bounding boxes on successfully detecting person object
            objects = [obj for obj in objects if obj[2] >= args.prob_threshold] # Draw only objects when probability more than specified threshold
            current_count = len(objects) 
            for obj in objects:
                xmin = int(obj[3] * width)
                ymin = int(obj[4] * height)
                xmax = int(obj[5] * width)
                ymax = int(obj[6] * height)
                class_id = int(obj[1])
                # Draw box and label\class_id
                color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                # det_label = labels_map[class_id] if labels_map else str(class_id)
                # det_label = str(class_id)
                det_label = "Person"
                cv2.putText(frame, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

            # Draw performance stats
            inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
                "Inference time: {:.3f} ms".format(det_time * 1000)
            render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1000)
            async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if is_async_mode else \
                "Async mode is off. Processing request {}".format(cur_request_id)

            cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            cv2.putText(frame, render_time_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            cv2.putText(frame, async_mode_message, (10, int(height - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (10, 10, 200), 1)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            ### When person is detected in the video
            if current_count > last_count:
                duration_start = time()

            ## Calculating the duration a person spent on video
            if current_count < last_count and int(time() - duration_start) >=3:
                duration = int(time() - duration_start)
                if duration > 0:
                    # Publish messages to the MQTT server
                    client.publish("person/duration",
                                   json.dumps({"duration": duration + lagtime}))
                    total_count += 1
                else:
                    lagtime += 1
                    # log.warning(lagtime)

            ## Send the useful stats to MQTT
            client.publish("person", json.dumps({"total": total_count})) 
            client.publish("person", json.dumps({"count": current_count}))
            last_count = current_count

        # ### Display the frame in cv show ###
        # render_start = time()
        # if not args.no_show:
        #     cv2.imshow("Detection Results", frame)
        # render_end = time()
        # render_time = render_end - render_start

        ### Send the frame to the FFMPEG server ###
        render_start = time()        
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        render_end = time()
        render_time = render_end - render_start

        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
            frame = next_frame
            height, width = frame.shape[:2]

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    ### TODO: Disconnect from MQTT
    client.disconnect()


def main():
    args = get_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == "__main__":
    main()
