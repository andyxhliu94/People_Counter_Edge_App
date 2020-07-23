# People_Counter_Edge_App
People Counter Edge App

Using OpenVino Version R3 on a local Mac

Tested on CAM and .mp4 test file

Also Includes:
  1. MQTT server
  2. FFmpeg server
  3. React frontend

Model used:
  1. SSD_mobile_net_v2
  2. Yolo_v3

Command line input example:
python main_ssd.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m "./frozen_inference_graph.xml" -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm


