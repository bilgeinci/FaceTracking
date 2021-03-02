# Face-Detection-using-Modified-Yolov4-Tiny-on-Jetson-Nano

<b>Introduction</b>

<p>In this study, face detection application was performed using an improved Yolov4-Tiny algorithm on the Jetson Nano platform. There are 2 YOLO outputs in the classic Yolov4-Tiny algorithm. These outputs are generally good at detecting large objects in a picture. In order to detect smaller objects, 3 or more outputs are required, as in Yolov3 and Yolov4 algorithms. In this study, successful results were obtained with a Yolov4-Tiny architecture with 3 prediction layers.</p>

<p>Within the scope of the study, not only face detection but also face tracking was performed. Deep Sort algorithm [2], which is a successful algorithm, is used as the tracking algorithm.  In this study, the procedures in Reference [3] are used in the application of the Deep Sort algorithm.</p>

<b>The structure of the developed system</b>
The flow chart of the developed system is shown in Figure 1. This method reads the frame in real time from the camera and transmits this frame to the algorithm for face detection. After the detection method, if there is a face in the picture, the detection boxes will be created and displayed. Then, these boxes are sent to the DeepSort algorithm to find tracking boxes, center point locations and IDs of detected faces. This process repeats until the next frame is gone.


![alt_text](https://github.com/bilgeinci/FaceTracking/blob/main/Images/Block-1.png)

Figure 1. The flowchart of the developed face detection and tracking system

<b>Yolov4-tiny with 3 Yolo-Layers</b>
The Yolov4-tiny method is designed based on the Yolov4 method to have a faster object detection speed. The Yolov4-tiny method uses the CSPDarknet53-Tiny network as the backbone network instead of the CSPDarknet53 network used in the Yolov4 method. The CSPDarknet53-tiny network uses the CSPBlock module in the cross-stage partial network instead of the ResBlock module in the Residual Network.

There are 2 prediction layers in the original Yolov4-Tiny architecture. These have 13x13xN and 26x26xN dimensions. These two outputs are insufficient for detecting small objects. In this study, a Yolov4-Tiny architecture with 3 outputs is used instead of the existing architecture to obtain a 56x56xN output. The parts highlighted in red in Figure 2 represent processes that are different from the existing Yolov4-Tiny structure.


![alt text](https://github.com/bilgeinci/FaceTracking/blob/main/Images/Block-2.png)

Figure 2. Yolov4-Tiny Architecture with 3 prediction layers

<b>Dataset</b>
In this study, WiderFace data set was used to create face detection model. The Wider Face dataset [4] is the most challenging public face detection dataset mainly due to the wide variety of face scales, poses, occlusions, expressions, illuminations and faces with heavily makeup. It has 32,203 images and nearly 400k annotated faces. This dataset randomly select 40%/10%/50% data as training, validation and testing sets. There are 50% faces with small scales (between 10-50 pixels), 43% faces with medium scales (between 50-300 pixels) and 7% faces with large scales (over 300 pixels) in the dataset [4]. There are three subsets with different difficulty: ‘Easy’, ‘Medium’, ‘Hard’. 

<b>System Hardware</b>
The developed system runs on Jetson Nano. A camera with a Sony IMX219 sensor is used as the camera module. The hardware structure can be seen in Figure 3.

![alt text](https://github.com/bilgeinci/FaceTracking/blob/main/Images/Block-4.png)

Figure 3. System Hardware

## Demo

Support video and webcam demo for now

Make sure everything is settled down
   - Yolov4-tiny-416 cfg and weights files
   - demo video you want to test on

Support 

1. onboard camera webcam 
2. Video track

- Webcam demo - onboard camera

  ```shell

  python run_tracker.py 
 
  ```

- Video demo

  ```shell
  
  python run_tracker.py --input video/your_test.mp4

  ```

Youtube Linki - Video için

Youtube Linki - Kamera için

------
## Speed

Whole process time from read image to finished deepsort (include every img preprocess and postprocess)

| Backbone                                      | detection + tracking | FPS(detection + tracking) |
| :---------------------------------------------| ---------------------| ------------------------- |
| Yolov4-tiny-416                               | 450ms                | 1.5 ~ 2                   |   
| Yolov4-tiny-416 (with 3 Yolo predicted layer) | 500ms                | 2   ~ 2.5                 |

------

<b>References</b>


1.	Alexey Bochkovskiy. Darknet: Open Source Neural Networks in Python. 2020. Available online: https://github.com/AlexeyAB/darknet (accessed on 2 November 2020) 
2.	Punn, N. S., Sonbhadra, S. K., & Agarwal, S. (2020). Monitoring COVID-19 social distancing with person detection and tracking via fine-tuned YOLO v3 and Deepsort techniques. arXiv preprint arXiv:2005.01385.
3.	https://github.com/theAIGuysCode/yolov3_deepsort
4.	Yang, S., Luo, P., Loy, C. C., & Tang, X. (2016). Wider face: A face detection benchmark. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 5525-5533).
5.	Toan, N. H., Nu-ri, S., Gwang-Hyun, Y., Gyeong-Ju, K., Woo-Young, K., & Jin-Young, K. (2020). Deep learning-based defective product classification system for smart factory.

