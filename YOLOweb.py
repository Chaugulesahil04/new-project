from ultralytics import YOLO
import cv2
import os
import time
from datetime import datetime
import mysql.connector
from mysql.connector import Error
import io

# Establish MySQL connection
def connect_to_mysql():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='rcnn',
            user='root',
            password=''
        )
        if connection.is_connected():
            print("Connected to MySQL database")
            return connection
    except Error as e:
        print("Error connecting to MySQL database:", e)
        return None

# Create tables if not exists
def create_tables(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS object_reports (
                id INT AUTO_INCREMENT PRIMARY KEY,
                object_name VARCHAR(255),
                detection_time DATETIME
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id INT AUTO_INCREMENT PRIMARY KEY,
                object_name VARCHAR(255),
                snapshot BLOB,
                snapshot_time DATETIME
            )
        """)
        connection.commit()
        print("Tables created successfully")
    except Error as e:
        print("Error creating tables:", e)

# Insert report into database
def insert_report(connection, object_name, detection_time):
    try:
        cursor = connection.cursor()
        query = "INSERT INTO object_reports (object_name, detection_time) VALUES (%s, %s)"
        cursor.execute(query, (object_name, detection_time))
        connection.commit()
        print("Report inserted into database")
    except Error as e:
        print("Error inserting report into database:", e)

# Insert snapshot into database
def insert_snapshot(connection, object_name, snapshot, snapshot_time):
    try:
        cursor = connection.cursor()
        query = "INSERT INTO snapshots (object_name, snapshot, snapshot_time) VALUES (%s, %s, %s)"
        cursor.execute(query, (object_name, snapshot, snapshot_time))
        connection.commit()
        print("Snapshot inserted into database")
    except Error as e:
        print("Error inserting snapshot into database:", e)

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Model
model = YOLO("yolov8n.pt")
# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Path to save snapshots
save_path = "snapshots"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Take time limit from user
time_limit = int(input("Enter the time limit for running the webcam (in seconds): "))

# Record start time
start_time = time.time()

# Connect to MySQL database
connection = connect_to_mysql()
if connection is None:
    exit()

# Create tables if not exists
create_tables(connection)

while True:
    if time.time() - start_time > time_limit:
        print("Time limit reached.")
        break

    success, img = cap.read()
    results = model(img, stream=True)

    # Extract current suspicious objects
    current_suspicious_objects = [(classNames[int(box.cls[0])], tuple(map(int, box.xyxy[0]))) for r in results for box in r.boxes
                                   if classNames[int(box.cls[0])] in ["cell phone", "laptop", "book", "clock", "bottle"]]

    # Capture snapshots of new suspicious objects
    for obj_name, (x1, y1, x2, y2) in current_suspicious_objects:
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # Save snapshot with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(save_path, f"{obj_name}_{timestamp}.jpg")
        cv2.imwrite(filename, img)
        print(f"Snapshot saved as: {filename}")

        # Convert image to binary data
        with open(filename, 'rb') as file:
            binary_data = file.read()

        # Insert snapshot into database
        insert_snapshot(connection, obj_name, binary_data, datetime.now())

        # Add captured object to the set
        insert_report(connection, obj_name, datetime.now())

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == ord('q'):  # Quit the program
        break

cap.release()
cv2.destroyAllWindows()
