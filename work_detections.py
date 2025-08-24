import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
import time
from math import hypot, acos, degrees
from scipy.signal import savgol_filter

# Load YOLOv8 model (pretrained)
model = YOLO("yolov8n.pt")

# Initialize Face Mesh from Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True)

# Eye aspect ratio function for blink detection
def eye_aspect_ratio(landmarks, left_ids, right_ids):
    def get_ear(points):
        A = hypot(points[1][0] - points[5][0], points[1][1] - points[5][1])
        B = hypot(points[2][0] - points[4][0], points[2][1] - points[4][1])
        C = hypot(points[0][0] - points[3][0], points[0][1] - points[3][1])
        return (A + B) / (2.0 * C)

    left = [landmarks[i] for i in left_ids]
    right = [landmarks[i] for i in right_ids]
    return (get_ear(left) + get_ear(right)) / 2.0

# Vector angle function using cosine similarity
def angle_between_vectors(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = hypot(v1[0], v1[1])
    mag2 = hypot(v2[0], v2[1])
    if mag1 * mag2 == 0:
        return 180
    return degrees(acos(dot / (mag1 * mag2)))

# Smoothing function for landmarks
def smooth_landmarks(landmark_history):
    if len(landmark_history) < 5:
        return landmark_history[-1]
    landmark_array = np.array(landmark_history[-5:])
    smoothed = savgol_filter(landmark_array, window_length=5, polyorder=2, axis=0)
    return smoothed[-1]

# Start webcam
# Replace with your IP
cap = cv2.VideoCapture(0)

face_states = {}
face_landmark_history = {}
face_id_counter = 0


def generate_face_id():
    global face_id_counter
    face_id_counter += 1
    return face_id_counter


def iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb)
    detections = model(frame_resized)[0]

    laptops = []
    for box in detections.boxes:
        cls = int(box.cls[0])
        if model.names[cls] == 'laptop':
            laptops.append(box.xyxy[0].cpu().numpy())

    if face_results.multi_face_landmarks:
        for i, landmarks in enumerate(face_results.multi_face_landmarks):
            h, w, _ = frame_resized.shape
            points = [(int(pt.x * w), int(pt.y * h)) for pt in landmarks.landmark]
            nose = points[1]
            left_eye = points[33]
            right_eye = points[263]
            eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

            face_box = [min(left_eye[0], right_eye[0]) - 20, min(nose[1] - 40, h), max(left_eye[0], right_eye[0]) + 20,
                        nose[1] + 40]
            matched_id = None

            for fid, data in face_states.items():
                if iou(face_box, data['bbox']) > 0.5:
                    matched_id = fid
                    break

            if matched_id is None:
                matched_id = generate_face_id()
                face_states[matched_id] = {
                    'bbox': face_box,
                    'total_not_working': 0,
                    'not_working_start': None,
                    'closed_eyes': 0,
                    'last_laptop': None
                }
                face_landmark_history[matched_id] = []

            face_states[matched_id]['bbox'] = face_box

            face_landmark_history[matched_id].append(points)
            smoothed_points = smooth_landmarks(face_landmark_history[matched_id])
            smoothed_nose = smoothed_points[1]
            smoothed_eye_center = ((smoothed_points[33][0] + smoothed_points[263][0]) // 2,
                                   (smoothed_points[33][1] + smoothed_points[263][1]) // 2)

            # Find nearest laptop center
            laptop_id = None
            min_dist = float('inf')
            for idx, lap in enumerate(laptops):
                lx = int((lap[0] + lap[2]) / 2)
                ly = int((lap[1] + lap[3]) / 2)
                dist = hypot(smoothed_nose[0] - lx, smoothed_nose[1] - ly)
                if dist < min_dist:
                    min_dist = dist
                    laptop_id = idx

            assigned_laptop = laptops[laptop_id] if laptop_id is not None and laptop_id < len(laptops) else None
            face_states[matched_id]['last_laptop'] = assigned_laptop

            direction_vector = (smoothed_nose[0] - smoothed_eye_center[0], smoothed_nose[1] - smoothed_eye_center[1])
            laptop_direction = "Unknown"
            gaze_accuracy = 0

            if assigned_laptop is not None:
                lx = int((assigned_laptop[0] + assigned_laptop[2]) / 2)
                ly = int((assigned_laptop[1] + assigned_laptop[3]) / 2)
                target_vector = (lx - smoothed_eye_center[0], ly - smoothed_eye_center[1])
                angle = angle_between_vectors(direction_vector, target_vector)
                gaze_accuracy = max(0, 100 - (angle / 90 * 100))

                if angle < 30:
                    laptop_direction = "Working"
                else:
                    laptop_direction = "Not Working"

            EAR = eye_aspect_ratio(points, [362, 385, 387, 263, 373, 380], [33, 160, 158, 133, 153, 144])
            eyes_closed = EAR < 0.2

            if laptop_direction == "Not Working" or eyes_closed:
                if face_states[matched_id]['not_working_start'] is None:
                    face_states[matched_id]['not_working_start'] = time.time()
            else:
                if face_states[matched_id]['not_working_start'] is not None:
                    elapsed = time.time() - face_states[matched_id]['not_working_start']
                    face_states[matched_id]['total_not_working'] += elapsed
                    face_states[matched_id]['not_working_start'] = None

            color = (0, 255, 0) if laptop_direction == "Working" and not eyes_closed else (0, 0, 255)
            x1, y1, x2, y2 = face_box
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_resized, f"ID: {matched_id}", (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame_resized, laptop_direction + (" + Eyes Closed" if eyes_closed else ""), (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame_resized, f"Gaze Acc: {int(gaze_accuracy)}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 0), 2)
            total_nw = face_states[matched_id]['total_not_working']
            if face_states[matched_id]['not_working_start']:
                total_nw += time.time() - face_states[matched_id]['not_working_start']
            cv2.putText(frame_resized, f"NW Time: {int(total_nw)}s", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    for i, lap in enumerate(laptops):
        lx1, ly1, lx2, ly2 = map(int, lap)
        cv2.rectangle(frame_resized, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2)
        cv2.putText(frame_resized, f"Laptop {i}", (lx1, ly1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Multi-user Gaze + Laptop Detection", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
