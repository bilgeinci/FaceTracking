# Importing the libraries
import cv2
import numpy as np
import sys
import os
from deepsort.tracker import DeepTracker
from deepsort import nn_matching
from deepsort import generate_detections
from deepsort.detection import Detection
from deepsort import preprocessing

import time

class FaceDetection():
    def __init__(self):
        self.nms_max_overlap = 1.0
        model_filename = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'),
                                      'mars-small128.pb')
        self.encoder = generate_detections.create_box_encoder(model_filename, batch_size=1)

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.3)
        self.tracker = DeepTracker(metric, max_iou_distance=0.9, max_age=5)
        self.memory = {}

    def start(self, input_file):
        reader = cv2.VideoCapture(input_file)

        fps = reader.get(cv2.CAP_PROP_FPS)
        frame_width = int(reader.get(3))
        frame_height = int(reader.get(4))

        output_file = "./video/output.mp4"
        writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 15, (frame_width, frame_height))

        ret, frame = reader.read()
        if not ret:
            print("Unknown command")
            sys.exit(2)

        net, output_layers, classes = self.load_model()

        fps = 0.0
        tic = time.time()
        while True:
            ret, frame = reader.read()
            boxes, class_ids = self.detect(net, output_layers, frame, confidence_threshold=0.2)

            features = self.encoder(frame, boxes)
            dets = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]

            boxes = np.array([d.tlwh for d in dets])
            scores = np.array([d.confidence for d in dets])
            indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
            detections = [dets[i] for i in indices]

            self.tracker.predict()
            self.tracker.update(detections)

            self.draw_tracks(frame)

            fpsText = "FPS: {}".format(round(fps, 2))
            #cv2.putText(frame, fpsText, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            writer.write(frame)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1)

            toc = time.time()
            fpsCurr = 1.0 / (toc - tic)
            fps = fpsCurr if fps == 0.0 else (fps * 0.95 + fpsCurr * 0.05)
            tic = toc

            if key == ord("q"):
                break

        reader.release()
        writer.release()
        cv2.destroyAllWindows()

    def load_model(self):
        net = cv2.dnn.readNetFromDarknet("./data/3l.cfg", "./data/3l.weights")
        with open("./data/obj.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return net, output_layers, classes

    # Funcion to detect objects
    def detect(self, net, output_layers, img, confidence_threshold):
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle Coordinates
                    x = center_x - w / 2
                    y = center_y - h / 2
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Applying Non-Max Suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

        boxes = [boxes[i[0]] for i in indexes]
        class_ids = [class_ids[i[0]] for i in indexes]
        return boxes, class_ids

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

    input_file = 0  # input video file location
    if len(sys.argv) > 1:
        if sys.argv[1] == "camera":
            input_file = 0
        else:
            input_file = sys.argv[1]
    else:
        print("usage: %s video path|camera" % sys.argv[0])
        sys.exit(2)

    faceDetection.start(input_file)


