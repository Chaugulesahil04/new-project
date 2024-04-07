'''import ultralytics
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
        database="sample",
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
detect_pose_and_objects(input_source)'''


import cv2
import numpy as np
import os
import time
import mysql.connector

from ultralytics import YOLO
# Load Mask RCNN model
net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
                                    "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")


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


def connect_to_database():
    # Connect to the database
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="sample"
    )
    return conn

def create_pose_data_table(cursor):
    # Create a table for storing pose data if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS pose_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        binary_output TEXT
    )
    """
    cursor.execute(create_table_query)

def create_video_data_table(cursor):
    # Create a table for storing video data if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS video_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        video_data LONGBLOB
    )
    """
    cursor.execute(create_table_query)

def insert_pose_data(conn, binary_output):
    # Insert pose data into the database
    cursor = conn.cursor()
    insert_query = "INSERT INTO pose_data (binary_output) VALUES (%s)"
    cursor.execute(insert_query, (binary_output,))
    conn.commit()
    cursor.close()

def insert_video_data(conn, video_path):
    # Read video file and insert its binary data into the database
    cursor = conn.cursor()
    with open(video_path, "rb") as f:
        video_data = f.read()
    insert_query = "INSERT INTO video_data (video_data) VALUES (%s)"
    cursor.execute(insert_query, (video_data,))
    conn.commit()
    cursor.close()

def detect_pose(input_source, output_folder, conn):
    # Initialize OpenPose model
    net = cv2.dnn.readNetFromTensorflow("dnn/graph_opt.pb")

    # Open webcam
    cap = cv2.VideoCapture(input_source)
    cap.set(3, 640)
    cap.set(4, 480)

    # Create a directory to store captured images if it does not exist
    captured_dir = "captured"
    if not os.path.exists(captured_dir):
        os.makedirs(captured_dir)

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

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

        # Mask RCNN detection
        blob = cv2.dnn.blobFromImage(img, swapRB=True)
        net.setInput(blob)
        out_net = net.forward()
        out_net = out_net[:, :19, :, :]  

        # Process OpenPose detections
        detected_parts = [1 if cv2.minMaxLoc(out_net[0, i, :, :])[1] > 0.2 else 0 for i in range(19)]
        binary_output = ''.join(map(str, detected_parts))
        print("Binary output:", binary_output)
        insert_pose_data(conn, binary_output)

        frameWidth = img.shape[1]
        frameHeight = img.shape[0]

        for i in range(19):
            heatMap = out_net[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out_net.shape[3]
            y = (frameHeight * point[1]) / out_net.shape[2]
            cv2.circle(img, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        
        # YOLOv5 detection
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].int()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Capture image if class is not 'person'
                if yolo_classes[int(box.cls[0])] != 'person':
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.imwrite(os.path.join(captured_dir, f"yolov5_detection_{count}.jpg"), img)
                    count += 1  # Increment image count

        # Write the frame to the output video file
        out_video.write(img)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    # Release resources
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

# Load YOLOv5 model
model = YOLO("yolov8n.pt")

# Connect to the MySQL database
conn = connect_to_database()

# Create a cursor
cursor = conn.cursor()

# Create tables for storing pose data and video data if they don't exist
create_pose_data_table(cursor)
create_video_data_table(cursor)

# Define the input source (0 for webcam)
input_source = 0

# Define the output folder
output_folder = "output_videos"
os.makedirs(output_folder, exist_ok=True)

# Call the function to detect pose and retrieve video file path and binary outputs
detect_pose(input_source, output_folder, conn)

# Insert video file path into the database
insert_video_data(conn, 'output.avi')

# Commit changes and close the cursor and connection
conn.commit()
cursor.close()
conn.close()


