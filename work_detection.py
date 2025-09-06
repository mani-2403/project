import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time
import math

# Load YOLOv8 model trained on COCO dataset
model = YOLO("yolov8n.pt")  # Can detect 'person' and 'laptop'

# Mediapipe face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, refine_landmarks=True)

# Eye aspect ratio threshold for detecting closed eyes
EYE_AR_THRESH = 0.20

def eye_aspect_ratio(landmarks, eye_indices):
    p1 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p2 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p5 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p6 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])

    vertical1 = np.linalg.norm(p2 - p4)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p6)

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def is_facing_laptop(nose_center, laptop_center, threshold=100):
    dx = nose_center[0] - laptop_center[0]
    dy = nose_center[1] - laptop_center[1]
    distance = math.sqrt(dx**2 + dy**2)
    return distance < threshold

 # Replace with your IP
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb)
    results = model(frame)[0]

    persons = []
    laptops = []

    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if cls == 0:
            persons.append(((x1, y1, x2, y2), get_center((x1, y1, x2, y2))))
        elif cls == 63:
            laptops.append(((x1, y1, x2, y2), get_center((x1, y1, x2, y2))))

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            h, w, _ = frame.shape
            nose = face_landmarks.landmark[1]
            nose_center = (int(nose.x * w), int(nose.y * h))

            left_ear = eye_aspect_ratio(face_landmarks.landmark, [33, 160, 158, 133, 153, 144])
            right_ear = eye_aspect_ratio(face_landmarks.landmark, [362, 385, 387, 263, 373, 380])
            avg_ear = (left_ear + right_ear) / 2
            eyes_open = avg_ear > EYE_AR_THRESH

            closest_laptop = None
            min_dist = float('inf')
            for laptop_box, laptop_center in laptops:
                dist = math.hypot(nose_center[0] - laptop_center[0], nose_center[1] - laptop_center[1])
                if dist < min_dist:
                    min_dist = dist
                    closest_laptop = (laptop_box, laptop_center)

            if closest_laptop and is_facing_laptop(nose_center, closest_laptop[1], threshold=150):
                if eyes_open:
                    status = "Working"
                    color = (0, 255, 0)
                else:
                    status = "Not Working (Eyes Closed)"
                    color = (0, 165, 255)
            else:
                status = "Not Working"
                color = (0, 0, 255)

            cv2.putText(frame, status, (nose_center[0], nose_center[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, nose_center, 5, color, -1)

    for (x1, y1, x2, y2), _ in persons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
    for (x1, y1, x2, y2), _ in laptops:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

    cv2.imshow("Laptop Attention Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
