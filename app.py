from flask import Flask, Response, render_template
import cv2
import time
from ultralytics import YOLO
import face_recognition  # ← NAYA ADD
import os                # ← NAYA ADD
import numpy as np       # ← NAYA ADD
from datetime import datetime  # ← NAYA ADD (optional for snapshots)

app = Flask(__name__)

# ================= YOLO PERSON MODEL =================
person_model = YOLO("yolov8n.pt")
print("YOLO PERSON MODEL LOADED")

# ================= FACE CASCADE =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
print("Face cascade empty:", face_cascade.empty())
print("FACE CASCADE LOADED")

# ================= KNOWN FACES LOAD =================  ← YE PURA NAYA SECTION
known_face_encodings = []
known_face_names = []

known_faces_dir = "known_faces"  # folder naam

if os.path.exists(known_faces_dir):
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:  # agar face mila
                known_face_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0].capitalize()  # filename se naam
                known_face_names.append(name)
                print(f"Loaded known face: {name}")
            else:
                print(f"No face found in {filename}")
else:
    print("known_faces folder nahi mila! Bana lo aur photos daalo.")

print(f"Total known faces loaded: {len(known_face_names)}")

# ================= CAMERA =================
cap = None
for i in range(3):
    temp = cv2.VideoCapture(i)
    if temp.isOpened():
        cap = temp
        print(f"Camera opened at index {i}")
        break

if cap is None:
    print("❌ Camera not found")
    exit()

cap.set(3, 1280)
cap.set(4, 720)

# ================= STREAM =================
def gen():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # -------- PERSON DETECTION (YOLO) --------
        results = person_model(frame, classes=[0], conf=0.5)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
                cv2.putText(frame, "Person", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # -------- FACE DETECTION + RECOGNITION --------
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # face_recognition ke liye RGB chahiye
        
        # Pehle face locations find karo (face_recognition se – zyada accurate)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            color = (0, 0, 255)  # red

            # Best match find karo
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                    name = known_face_names[best_match_index]
                    color = (0, 255, 0)  # green

            # Box aur naam daalo
            cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
            cv2.putText(frame, name, (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

            # Optional: Unknown person ki photo save karo
            if name == "Unknown":
                os.makedirs("snapshots", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"snapshots/unknown_{timestamp}.jpg", frame)
                print(f"Unknown person detected! Photo saved: unknown_{timestamp}.jpg")

        # -------- TIME --------
        cv2.putText(frame, time.strftime("%H:%M:%S"),
                    (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ================= ROUTES =================
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video')
def video():
    return Response(gen(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

# ================= RUN =================
if __name__ == '__main__':
    print("OPEN: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)