import cv2
import numpy as np
import sys
import os
from deepsort.tracker import DeepTracker
from deepsort import nn_matching
from deepsort import generate_detections
from deepsort.detection import Detection
from deepsort import preprocessing

import argparse
import time
from yolo_with_plugins import TrtYOLO

import pycuda.autoinit

class FaceDetection():
    def __init__(self):
        self.nms_max_overlap = 1.0
        model_filename = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'),
                                      'mars-small128.pb')
        self.encoder = generate_detections.create_box_encoder(model_filename, batch_size=1)

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.3)
        self.tracker = DeepTracker(metric, max_iou_distance=0.7, max_age=8)
        self.model = TrtYOLO("yolov4-tiny-416", (416, 416), 1, False)
        self.memory = {}

    def start(self, input_file):
        reader = cv2.VideoCapture(input_file)
        ret, frame = reader.read()
        if not ret:
            print("No input")
            sys.exit(2)

        fps = 0.0
        tic = time.time()
        while True:
            ret, frame = reader.read()
            
            self.detection_tracking_unit(frame, 0.5)
            self.draw_tracks(frame)

            fpsText = "FPS: {}".format(round(fps, 2))
            #cv2.putText(frame, fpsText, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1)

            toc = time.time()
            fpsCurr = 1.0 / (toc - tic)
            fps = fpsCurr if fps == 0.0 else (fps * 0.95 + fpsCurr * 0.05)
            tic = toc

            if key == ord("q"):
                break

        reader.release()
        cv2.destroyAllWindows()

    def detection_tracking_unit(self, image, conf_th):
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dets = []

        bboxes, confs, clss = self.model.detect(image, conf_th) 
        for box in bboxes:
            dets.append(np.array([int(box[0]), int(box[1]), int(box[2]-box[0]+10), int(box[3]-box[1]+10)]))

        features = self.encoder(image1, dets)
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(dets, features)]

        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, 1.0, scores)
        detections = [detections[i] for i in indices]

        self.tracker.predict()
        self.tracker.update(detections)

        del image1
        return dets

    def draw_tracks(self, frame):
        boxes = []
        indexIDs = []
        previous = self.memory.copy()
        self.memory = {}

        i = int(0)
        for track in self.tracker.tracks:
            if not track.is_confirmed():
                continue

            bbox = track.to_tlbr()
            boxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
            indexIDs.append(track.track_id)
            self.memory[indexIDs[-1]] = boxes[-1]

        for box in boxes:
            if indexIDs[i] in previous:
                # extract the bounding box coordinates
                (tx, ty) = (int(box[0]), int(box[1]))
                (bx, by) = (int(box[2]), int(box[3]))

                cv2.rectangle(frame, (tx, ty), (bx, by), (0, 255, 0), 2)
                text = "ID: {}".format(indexIDs[i])
                cv2.putText(frame, text, (int(tx), int(ty - 15)), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)

            i += 1



if __name__ == '__main__':
    faceDetection = FaceDetection()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='camera', help=('video/your_test.mp4'))
    args = parser.parse_args()

    input_file = 0  # input video file location
    if args.input == "camera":
        input_file = "nvarguscamerasrc sensor_mode=0 ! video/x-raw(memory:NVMM), format=NV12, width=3820, height=2464, framerate=21/1 ! nvvidconv ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    else:
        input_file = args.input

    faceDetection.start(input_file)


