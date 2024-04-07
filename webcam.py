'''import cv2
import numpy as np
from ultralytics import YOLO
import math 

# Load Mask RCNN model
net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
                                    "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

# Load YOLOv5 model
model = YOLO("yolov8n.pt")

# Generate random colors
colors = np.random.randint(0, 255, (80, 3))

# Object classes for Mask RCNN
mask_rcnn_classes = {
    0: 'person',
    9: 'pen',
    40: 'bottle',
    # Add more classes as needed for objects in an examination hall
}

# Object classes for YOLOv5
yolo_classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"]

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 420)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame from webcam")
        break

    # Mask RCNN detection
    blob = cv2.dnn.blobFromImage(img, swapRB=True)
    net.setInput(blob)
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    detection_count = boxes.shape[2]

    # Process Mask RCNN detections
    for i in range(detection_count):
        box = boxes[0, 0, i]
        class_id = int(box[1])
        score = box[2]
        if score < 0.5:
            continue

        x = int(box[3] * img.shape[1])
        y = int(box[4] * img.shape[0])
        x2 = int(box[5] * img.shape[1])
        y2 = int(box[6] * img.shape[0])

        # Draw bounding box and label for Mask RCNN detections
        cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, mask_rcnn_classes.get(class_id, 'unknown'), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # YOLOv5 detection
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box and label for YOLOv5 detections
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, yolo_classes[int(box.cls[0])], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''



# detect image except person and capture the image 
'''import cv2
import numpy as np
from ultralytics import YOLO

# Load Mask RCNN model
net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
                                    "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

# Load YOLOv5 model
model = YOLO("yolov8n.pt")

# Object classes for Mask RCNN
mask_rcnn_classes = {
    0: 'person',
    9: 'pen',
    40: 'bottle',
    # Add more classes as needed for objects in an examination hall
}

# Object classes for YOLOv5
yolo_classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"]

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 420)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame from webcam")
        break

    # Mask RCNN detection
    blob = cv2.dnn.blobFromImage(img, swapRB=True)
    net.setInput(blob)
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    detection_count = boxes.shape[2]

    # Process Mask RCNN detections
    for i in range(detection_count):
        box = boxes[0, 0, i]
        class_id = int(box[1])
        score = box[2]
        if score < 0.5:
            continue

        x = int(box[3] * img.shape[1])
        y = int(box[4] * img.shape[0])
        x2 = int(box[5] * img.shape[1])
        y2 = int(box[6] * img.shape[0])

        # Draw bounding box and label for Mask RCNN detections
        cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, mask_rcnn_classes.get(class_id, 'unknown'), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Capture image if class is not 'person'
        if mask_rcnn_classes.get(class_id, 'unknown') != 'person':
            cv2.imwrite("mask_rcnn_detection.jpg", img)

    # YOLOv5 detection
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box and label for YOLOv5 detections
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, yolo_classes[int(box.cls[0])], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Capture image if class is not 'person'
            if yolo_classes[int(box.cls[0])] != 'person':
                cv2.imwrite("yolov5_detection.jpg", img)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''


#red boundary box

'''import cv2
import numpy as np
from ultralytics import YOLO

# Load Mask RCNN model
net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
                                    "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

# Load YOLOv5 model
model = YOLO("yolov8n.pt")

# Object classes for Mask RCNN
mask_rcnn_classes = {
    0: 'person',
    9: 'pen',
    40: 'bottle',
    # Add more classes as needed for objects in an examination hall
}

# Object classes for YOLOv5
yolo_classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"]

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 420)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame from webcam")
        break

    # Mask RCNN detection
    blob = cv2.dnn.blobFromImage(img, swapRB=True)
    net.setInput(blob)
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    detection_count = boxes.shape[2]

    # Process Mask RCNN detections
    for i in range(detection_count):
        box = boxes[0, 0, i]
        class_id = int(box[1])
        score = box[2]
        if score < 0.5:
            continue

        x = int(box[3] * img.shape[1])
        y = int(box[4] * img.shape[0])
        x2 = int(box[5] * img.shape[1])
        y2 = int(box[6] * img.shape[0])

        # Draw bounding box and label for Mask RCNN detections
        cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, mask_rcnn_classes.get(class_id, 'unknown'), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Capture image if class is not 'person'
        if mask_rcnn_classes.get(class_id, 'unknown') != 'person':
            cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite("mask_rcnn_detection.jpg", img)

    # YOLOv5 detection
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box and label for YOLOv5 detections
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, yolo_classes[int(box.cls[0])], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Capture image if class is not 'person'
            if yolo_classes[int(box.cls[0])] != 'person':
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.imwrite("yolov5_detection.jpg", img)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''

#captured one image
'''import cv2
import numpy as np
from ultralytics import YOLO

# Load Mask RCNN model
net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
                                    "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

# Load YOLOv5 model
model = YOLO("yolov8n.pt")

# Object classes for Mask RCNN
mask_rcnn_classes = {
    0: 'person',
    9: 'pen',
    40: 'bottle',
    # Add more classes as needed for objects in an examination hall
}

# Object classes for YOLOv5
yolo_classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"]

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 420)

# Initialize flag to capture only one snapshot for each non-person object detected
captured = False

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame from webcam")
        break

    # Mask RCNN detection
    blob = cv2.dnn.blobFromImage(img, swapRB=True)
    net.setInput(blob)
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    detection_count = boxes.shape[2]

    # Process Mask RCNN detections
    for i in range(detection_count):
        box = boxes[0, 0, i]
        class_id = int(box[1])
        score = box[2]
        if score < 0.5:
            continue

        x = int(box[3] * img.shape[1])
        y = int(box[4] * img.shape[0])
        x2 = int(box[5] * img.shape[1])
        y2 = int(box[6] * img.shape[0])

        # Capture image if class is not 'person'
        if mask_rcnn_classes.get(class_id, 'unknown') != 'person' and not captured:
            captured = True
            mask_img = np.zeros_like(img)
            cv2.rectangle(mask_img, (x, y), (x2, y2), (255, 255, 255), -1)
            masked_img = cv2.bitwise_and(img, mask_img)
            cv2.imwrite("mask_rcnn_detection.jpg", masked_img)

    # YOLOv5 detection
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Capture image if class is not 'person'
            if yolo_classes[int(box.cls[0])] != 'person' and not captured:
                captured = True
                mask_img = np.zeros_like(img)
                cv2.rectangle(mask_img, (x1, y1), (x2, y2), (255, 255, 255), -1)
                masked_img = cv2.bitwise_and(img, mask_img)
                cv2.imwrite("yolov5_detection.jpg", masked_img)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''
import ultralytics
import cv2
import numpy as np
import os
import time
import mysql.connector

def detect_pose_and_objects(input_source):
    # Load OpenPose model
    net_openpose = cv2.dnn.readNetFromTensorflow("dnn/graph_opt.pb")

    # Load Mask RCNN model
    net_mask_rcnn = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
                                        "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

    # Load YOLOv5 model
    model_yolo = ultralytics.YOLO("yolov8n.pt")

    # Object classes for Mask RCNN
    mask_rcnn_classes = {
        0: 'person',
        9: 'pen',
        40: 'bottle',
        # Add more classes as needed for objects in an examination hall
    }

    # Object classes for YOLOv5
    yolo_classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                    "teddy bear", "hair drier", "toothbrush"]

    # Create a directory to store captured images if it does not exist
    captured_dir = "captured"
    if not os.path.exists(captured_dir):
        os.makedirs(captured_dir)

    # Connect to MySQL database
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="rcnn",
        port="3306"
    )
    cursor = mydb.cursor()

    # Start webcam
    cap = cv2.VideoCapture(input_source)
    cap.set(3, 640)
    cap.set(4, 420)

    count = 1  # Initialize image count

    # Previous positions of nose, shoulders, and hips
    prev_nose_pos = None
    prev_shoulder_pos = None
    prev_hip_pos = None

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read frame from webcam")
            break

        frameWidth = img.shape[1]
        frameHeight = img.shape[0]

        # OpenPose detection
        net_openpose.setInput(cv2.dnn.blobFromImage(img, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out_net = net_openpose.forward()
        out_net = out_net[:, :19, :, :]  

        # Object detection with Mask RCNN
        blob = cv2.dnn.blobFromImage(img, swapRB=True)
        net_mask_rcnn.setInput(blob)
        boxes, masks = net_mask_rcnn.forward(["detection_out_final", "detection_masks"])
        detection_count = boxes.shape[2]

        # Object detection with YOLOv5
        results = model_yolo(img, stream=True)

        # Process detections
        for i in range(detection_count):
            box = boxes[0, 0, i]
            class_id = int(box[1])
            score = box[2]
            if score < 0.5:
                continue

            x = int(box[3] * frameWidth)
            y = int(box[4] * frameHeight)
            x2 = int(box[5] * frameWidth)
            y2 = int(box[6] * frameHeight)

            # Capture image if class is not 'person' for Mask RCNN
            if mask_rcnn_classes.get(class_id, 'unknown') != 'person':
                cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 2)
                image_path = os.path.join(captured_dir, f"mask_rcnn_detection_{count}.jpg")
                cv2.imwrite(image_path, img)
                with open(image_path, "rb") as f:
                    binary_data = f.read()
                    sql = "INSERT INTO images (image_data) VALUES (%s)"
                    cursor.execute(sql, (binary_data,))
                    mydb.commit()
                count += 1  # Increment image count

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].int()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Capture image if class is not 'person' for YOLOv5
                if yolo_classes[int(box.cls[0])] != 'person':
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    image_path = os.path.join(captured_dir, f"yolov5_detection_{count}.jpg")
                    cv2.imwrite(image_path, img)
                    with open(image_path, "rb") as f:
                        binary_data = f.read()
                        sql = "INSERT INTO images (image_data) VALUES (%s)"
                        cursor.execute(sql, (binary_data,))
                        mydb.commit()
                    count += 1  # Increment image count

        # Pose detection logic goes here
        # You can utilize the out_net from OpenPose model for pose detection

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Define the input source (0 for webcam)
input_source = 0

# Call the function to detect pose and objects
detect_pose_and_objects(input_source)


