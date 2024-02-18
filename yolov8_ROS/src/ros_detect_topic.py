#!/usr/bin/env python3

from collections import defaultdict

import cv2
import numpy as np

import rospy
from rostopic import get_topic_type
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import UInt8, String
from yolov8_ROS.msg import Yolo_Objects, Objects
from cv_bridge import CvBridge
import torch

import os
import sys
from pathlib import Path

# yolov8 path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "ultralytics"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  

from ultralytics import YOLO

class YoloV8_ROS():
    def __init__(self):
        rospy.Subscriber(source, CompressedImage, self.Callback)
        self.pub = rospy.Publisher("yolov8_pub", data_class=Yolo_Objects, queue_size = 10)
        self.weights = rospy.get_param("~weights") # model path [(ex) model.pt]

        #  Inference Arguments -> https://docs.ultralytics.com/modes/predict/#inference-sources 
        self.conf = rospy.get_param("~conf") # confidence threshold [(ex) 0.25]
        imgsz_h = rospy.get_param("~imgsz_h") # image size height [(ex) 640]
        imgsz_w = rospy.get_param("~imgsz_w") # image size width [(ex) 640]
        self.imgsz = (imgsz_h, imgsz_w)
        self.device = torch.device(rospy.get_param("~device")) # cuda device [(ex) 0 or 0/1/2/3 or cpu]

        # Load the YOLOv8 model
        self.model = model = YOLO(self.weights)

        # Store the track history
        self.track_history = defaultdict(lambda: [])
        
    def Callback(self, data):
        bridge = CvBridge()
        frame = bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        msg = Yolo_Objects()
        msg.header = data.header

        #cv2.namedWindow("YOLOv8 Tracking", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO) # allow window resize (Linux)
        
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = self.model.track(frame, persist=True, conf=self.conf, device=self.device)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh

        # Get Class
        Class = results[0].boxes.cls

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        if results[0].boxes.id is not None:

            track_ids = results[0].boxes.id.int().tolist()

            # Plot the tracks
            for box, cls, track_id in zip(boxes, Class, track_ids):
                x, y, w, h = box # x, y = center points
                track = self.track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 90:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 0, 255), thickness=10)
                
                cls = int(cls.numpy())
                x1 = int(x.numpy()) - int(w.numpy() / 2)
                x2 = int(x.numpy()) + int(w.numpy() / 2)
                y1 = int(y.numpy()) - int(h.numpy() / 2)
                y2 = int(y.numpy()) + int(h.numpy() / 2)

                msg.yolo_objects.append(Objects(cls, track_id, x1, x2, y1, y2))

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        cv2.waitKey(1) 

        self.pub.publish(msg)


def run():
    global source
    rospy.init_node("yolov8_ROS")
    source = rospy.get_param("~source") # topic name [(ex) /camera/image]
    detect = YoloV8_ROS()
    rospy.spin()

if __name__ == '__main__':
    run()