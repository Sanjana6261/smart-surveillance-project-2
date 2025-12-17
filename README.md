# Smart Surveillance Project 2 🚨👁️

An AI-powered **Smart Surveillance System** that performs **real-time face detection and recognition** using **YOLOv8**, **OpenCV**, and **Python**. The system identifies known individuals from a stored dataset and automatically captures snapshots of unknown persons for security monitoring.

---

## 🔍 Overview

This project uses deep learning–based object detection (YOLOv8) to detect faces from a live camera feed. Detected faces are matched against a pre-registered dataset. If the person is unknown, the system saves snapshots for later review. A simple web interface displays the live stream and results.

---

## ✨ Key Features

* Real-time face detection using **YOLOv8**
* Face recognition for **known persons**
* Automatic snapshot capture for **unknown persons**
* Live camera streaming
* Web-based UI
* Modular and scalable project structure

---

## 🛠️ Technologies Used

* **Python**
* **OpenCV**
* **YOLOv8**
* **Face Recognition**
* **Flask** (for web interface)
* **HTML/CSS**

---

## 📁 Project Structure

```
smart-surveillance-project-2/
│
├── app.py                  # Main application file
├── camera.py               # Camera & detection logic
├── requirements.txt        # Project dependencies
├── templates/              # HTML templates
├── known_faces/             # Dataset of known persons
├── snapshots/               # Captured images of unknown persons
├── yolov8n.pt               # YOLOv8 object detection model
├── yolov8n-face.pt          # YOLOv8 face detection model
└── .gitignore
```

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Sanjana6261/smart-surveillance-project-2.git
cd smart-surveillance-project-2
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
python app.py
```

Open your browser and follow the local URL shown in the terminal.

---

## 🎯 Use Cases

* Smart security systems
* Office and campus surveillance
* Unauthorized access detection
* AI-based monitoring applications

---

## 👩‍💻 Author

**Sanjana Kushwah**
GitHub: [https://github.com/Sanjana6261](https://github.com/Sanjana6261)

---


