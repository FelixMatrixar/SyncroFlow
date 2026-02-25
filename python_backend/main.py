# python_backend\main.py

import os
import json
import base64
import csv
import urllib.request
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2
import onnxruntime as ort
import requests
from collections import deque

from dotenv import load_dotenv
load_dotenv()  # This physically loads the variables from .env into os.environ

# OpenRouter API Key
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    print("⚠ WARNING: OPENROUTER_API_KEY not found in environment variables!")

# Create a local 'models' directory in the backend folder
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# YOLOv11 / YOLO-World Model Setup
YOLO_MODEL_PATH = MODELS_DIR / "yolov8s-worldv2.onnx"
YOLO_CLASSES_PATH = MODELS_DIR / "coco_classes.txt"

# COCO class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# YOLOv11 model (will be loaded on first use)
yolo_net = None

# YOLOv11-pose Model Setup
YOLO_POSE_MODEL_PATH = MODELS_DIR / "yolo11n-pose.onnx"

# COCO keypoint names (17 keypoints)
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Skeleton connections for drawing pose (pairs of keypoint indices)
POSE_SKELETON = [
    (0, 1), (0, 2),  # nose to eyes
    (1, 3), (2, 4),  # eyes to ears
    (0, 5), (0, 6),  # nose to shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10),  # right arm
    (5, 6),  # shoulders
    (5, 11), (6, 12),  # shoulders to hips
    (11, 12),  # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# YOLOv11-pose model (will be loaded on first use)
yolo_pose_net = None


class ExecutionSession:
    """
    Tracks motion state across frames for a single execution.
    Maintains frame history and calculates velocity to determine if objects are moving or stationary.
    """
    def __init__(self, execution_id: str, target_class: str = "boat"):
        self.execution_id = execution_id
        self.target_class = target_class
        
        # Frame history (store last 10 frames)
        self.frame_history = deque(maxlen=10)
        
        # Detection history with timestamps
        self.detection_history = deque(maxlen=30)  # ~6 seconds at 5 fps
        
        # Motion state tracking
        self.motion_state = "waiting"  # waiting, approaching, docking, stationary
        self.last_state_change = time.time()
        self.stationary_frames = 0
        self.stationary_threshold_frames = 10  # Must be stationary for 10 frames (2 seconds at 5fps)
        
        # Velocity tracking
        self.velocity_history = deque(maxlen=10)
        self.current_velocity = 0.0
        
        # Re-arming mechanism (for multiple dockings in same execution)
        self.consecutive_misses = 0
        self.miss_threshold = 15  # Reset after 15 frames (~3 seconds) without detection
        
        # Buffered stationary frame (for accurate analysis)
        self.buffered_frame = None
        self.buffered_detections = []
        self.buffered_timestamp = None
        
        # Thresholds (in normalized coordinates per second)
        self.APPROACHING_THRESHOLD = 0.02  # Moving > 2% per second
        self.STATIONARY_THRESHOLD = 0.005  # Moving < 0.5% per second
        
    def calculate_centroid(self, bbox):
        """Calculate center point of bounding box (bbox: [x%, y%, w%, h%])"""
        x, y, w, h = bbox
        center_x = x + w / 2
        center_y = y + h / 2
        return (center_x, center_y)
    
    def calculate_velocity(self, centroid1, centroid2, time_delta):
        """Calculate velocity between two centroids (in % per second)"""
        if time_delta <= 0:
            return 0.0
        
        dx = centroid2[0] - centroid1[0]
        dy = centroid2[1] - centroid1[1]
        distance = np.sqrt(dx**2 + dy**2)
        velocity = distance / time_delta
        return velocity
    
    def add_detection(self, detections: List[Dict], timestamp: float):
        """
        Add new detection and update motion state.
        Returns: (motion_state, velocity, should_analyze)
        """
        # Filter for target class
        target_detections = [d for d in detections if d.get("class") == self.target_class]
        
        if not target_detections:
            # No target object detected - increment miss counter
            self.consecutive_misses += 1
            self.detection_history.append({
                "timestamp": timestamp,
                "detected": False,
                "centroid": None,
                "bbox": None
            })
            
            # Re-arm after sustained absence (allows for multiple dockings per execution)
            if self.consecutive_misses >= self.miss_threshold:
                previous_state = self.motion_state
                self.motion_state = "waiting"
                self.stationary_frames = 0
                self.velocity_history.clear()
                self.current_velocity = 0.0
                self.detection_history.clear()  # Clear old detections for fresh velocity calculations
                if previous_state != "waiting":
                    print(f"🔄 [{self.execution_id}] Re-armed: target absent for {self.consecutive_misses} frames, state reset to waiting")
            
            return self.motion_state, 0.0, False
        
        # Reset miss counter when target is detected
        self.consecutive_misses = 0
        
        # Use the detection with highest confidence
        target = max(target_detections, key=lambda d: d.get("confidence", 0))
        bbox = target.get("bbox", [0, 0, 0, 0])
        centroid = self.calculate_centroid(bbox)
        
        # Add to history
        detection_record = {
            "timestamp": timestamp,
            "detected": True,
            "centroid": centroid,
            "bbox": bbox,
            "confidence": target.get("confidence", 0)
        }
        self.detection_history.append(detection_record)
        
        # Calculate velocity if we have previous detections
        if len(self.detection_history) >= 2:
            prev_record = None
            # Find most recent previous detection with centroid
            for i in range(len(self.detection_history) - 2, -1, -1):
                if self.detection_history[i]["detected"] and self.detection_history[i]["centroid"]:
                    prev_record = self.detection_history[i]
                    break
            
            if prev_record:
                time_delta = timestamp - prev_record["timestamp"]
                velocity = self.calculate_velocity(prev_record["centroid"], centroid, time_delta)
                self.velocity_history.append(velocity)
                
                # Use smoothed velocity (average of last 5 measurements)
                if len(self.velocity_history) >= 3:
                    self.current_velocity = np.mean(list(self.velocity_history)[-5:])
                else:
                    self.current_velocity = velocity
        
        # Determine motion state
        previous_state = self.motion_state
        
        if self.current_velocity > self.APPROACHING_THRESHOLD:
            self.motion_state = "approaching"
            self.stationary_frames = 0
        elif self.current_velocity < self.STATIONARY_THRESHOLD:
            self.stationary_frames += 1
            if self.stationary_frames >= self.stationary_threshold_frames:
                self.motion_state = "stationary"
            elif self.motion_state == "approaching":
                self.motion_state = "docking"
        else:
            # In between - consider it docking
            self.motion_state = "docking"
            self.stationary_frames = 0
        
        # Track state changes
        if previous_state != self.motion_state:
            self.last_state_change = timestamp
        
        # Should we trigger analysis?
        # Yes if we just became stationary
        should_analyze = (previous_state != "stationary" and self.motion_state == "stationary")
        
        # Clear buffered frame when re-arming (transitioning from stationary to non-stationary)
        if previous_state == "stationary" and self.motion_state != "stationary":
            self.buffered_frame = None
            self.buffered_detections = []
        
        return self.motion_state, self.current_velocity, should_analyze
    
    def get_state_summary(self):
        """Get current state summary for debugging"""
        return {
            "execution_id": self.execution_id,
            "target_class": self.target_class,
            "motion_state": self.motion_state,
            "current_velocity": round(self.current_velocity * 100, 2),  # Convert to % per second
            "stationary_frames": self.stationary_frames,
            "consecutive_misses": self.consecutive_misses,
            "detections_tracked": len(self.detection_history),
            "last_state_change": self.last_state_change
        }


# Global execution sessions storage
execution_sessions: Dict[str, ExecutionSession] = {}


def download_yolo_model():
    """Download or export YOLOv11 ONNX model if not present"""
    if not YOLO_MODEL_PATH.exists():
        # Try using Ultralytics to export the model first
        try:
            print("📥 Attempting to export YOLOv11n using Ultralytics...")
            from ultralytics import YOLOWorld
            # yolov8s-world.pt is the small, fast version of YOLO World
            model = YOLOWorld("yolov8s-worldv2.pt")
            
            # Optional: Set custom classes if you want it to look for specific things by default
            # model.set_classes(["person", "backpack", "laptop"])
            
            print("🔄 Exporting YOLO-World to ONNX format...")
            export_path = model.export(format="onnx", dynamic=True, simplify=True)

            # Move to expected location
            export_file = Path(export_path)
            if export_file.exists() and export_file != YOLO_MODEL_PATH:
                export_file.replace(YOLO_MODEL_PATH)

            if YOLO_MODEL_PATH.exists():
                size = YOLO_MODEL_PATH.stat().st_size
                print(
                    f"✓ Model exported successfully ({size/1024/1024:.1f}MB)")
                print(f"   Saved to: {YOLO_MODEL_PATH}")
                return
        except ImportError:
            print(
                "⚠ Ultralytics not available, falling back to downloading pre-exported model..."
            )
        except Exception as e:
            print(f"⚠ Export failed: {e}")
            print("   Falling back to downloading pre-exported model...")

        # Fallback: Download pre-exported model from Hugging Face
        print("📥 Downloading YOLOv11n model from Hugging Face...")
        model_urls = [
            "https://huggingface.co/giangndm/yolo11-onnx/resolve/main/yolo11n_640.onnx",
        ]

        downloaded = False
        for model_url in model_urls:
            try:
                print(f"   Trying: {model_url}")
                # Download with retry
                for attempt in range(3):
                    try:
                        urllib.request.urlretrieve(model_url, YOLO_MODEL_PATH)
                        # Verify download succeeded (check file size)
                        size = YOLO_MODEL_PATH.stat().st_size
                        if size > 1_000_000:  # At least 1MB
                            print(
                                f"   ✓ Model downloaded successfully ({size/1024/1024:.1f}MB)"
                            )
                            print(f"   Saved to: {YOLO_MODEL_PATH}")
                            downloaded = True
                            break
                        else:
                            print(
                                f"   ⚠ Downloaded file too small ({size} bytes), retrying..."
                            )
                            YOLO_MODEL_PATH.unlink()
                    except Exception as e:
                        if attempt < 2:
                            print(f"   Retry {attempt + 1}/3...")
                        else:
                            raise e
                if downloaded:
                    break
            except Exception as e:
                print(f"   ✗ Failed: {e}")
                continue

        if not downloaded:
            error_msg = (
                "\n❌ Failed to get YOLOv11n model automatically.\n"
                "Please try one of these options:\n"
                "Option 1: Install Ultralytics and run export script:\n"
                "   pip install ultralytics\n"
                "   python python_backend/export_yolo_model.py\n"
                "Option 2: Download manually:\n"
                "   Visit: https://huggingface.co/giangndm/yolo11-onnx/tree/main\n"
                "   Download yolo11n_640.onnx\n"
                "   Save as: /tmp/yolo11n.onnx\n"
                "Then restart the Python backend.\n")
            raise RuntimeError(error_msg)


def load_yolo_model():
    """Load YOLOv11 model with ONNX Runtime"""
    global yolo_net
    if yolo_net is None:
        download_yolo_model()
        # Use ONNX Runtime instead of OpenCV DNN for better compatibility
        yolo_net = ort.InferenceSession(str(YOLO_MODEL_PATH),
                                        providers=['CPUExecutionProvider'])
        print("✓ YOLOv11 model loaded successfully with ONNX Runtime")
    return yolo_net


def download_yolo_pose_model():
    """Download or export YOLOv11-pose ONNX model if not present"""
    if not YOLO_POSE_MODEL_PATH.exists():
        # Try using Ultralytics to export the pose model first
        try:
            print("📥 Attempting to export YOLOv11n-pose using Ultralytics...")
            from ultralytics import YOLOWorld
            # yolov8s-world.pt is the small, fast version of YOLO World
            model = YOLOWorld("yolov8s-worldv2.pt")
            
            # Optional: Set custom classes if you want it to look for specific things by default
            # model.set_classes(["person", "backpack", "laptop"])
            
            print("🔄 Exporting YOLO-World to ONNX format...")
            export_path = model.export(format="onnx", dynamic=True, simplify=True)

            # Move to expected location
            export_file = Path(export_path)
            if export_file.exists() and export_file != YOLO_POSE_MODEL_PATH:
                export_file.replace(YOLO_POSE_MODEL_PATH)

            if YOLO_POSE_MODEL_PATH.exists():
                size = YOLO_POSE_MODEL_PATH.stat().st_size
                print(
                    f"✓ Pose model exported successfully ({size/1024/1024:.1f}MB)")
                print(f"   Saved to: {YOLO_POSE_MODEL_PATH}")
                return
        except ImportError:
            print(
                "⚠ Ultralytics not available, falling back to downloading pre-exported model..."
            )
        except Exception as e:
            print(f"⚠ Export failed: {e}")
            print("   Falling back to downloading pre-exported model...")

        # Fallback: Download pre-exported model from Hugging Face
        print("📥 Downloading YOLOv11n-pose model from Hugging Face...")
        model_urls = [
            "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11n-pose.onnx",
        ]

        downloaded = False
        for model_url in model_urls:
            try:
                print(f"   Trying: {model_url}")
                # Download with retry
                for attempt in range(3):
                    try:
                        urllib.request.urlretrieve(model_url, YOLO_POSE_MODEL_PATH)
                        # Verify download succeeded (check file size)
                        size = YOLO_POSE_MODEL_PATH.stat().st_size
                        if size > 1_000_000:  # At least 1MB
                            print(
                                f"   ✓ Pose model downloaded successfully ({size/1024/1024:.1f}MB)"
                            )
                            print(f"   Saved to: {YOLO_POSE_MODEL_PATH}")
                            downloaded = True
                            break
                        else:
                            print(
                                f"   ✗ Download incomplete (only {size} bytes)"
                            )
                            YOLO_POSE_MODEL_PATH.unlink()
                    except Exception as e:
                        if attempt < 2:
                            print(f"   Retry {attempt + 1}/3...")
                        else:
                            raise e
                if downloaded:
                    break
            except Exception as e:
                print(f"   ✗ Failed: {e}")
                continue

        if not downloaded:
            error_msg = (
                "\n❌ Failed to get YOLOv11n-pose model automatically.\n"
                "Please try one of these options:\n"
                "Option 1: Install Ultralytics and run export:\n"
                "   pip install ultralytics\n"
                "   python -c 'from ultralytics import YOLO; YOLO(\"yolo11n-pose.pt\").export(format=\"onnx\")'\n"
                "Option 2: Download manually:\n"
                "   Visit: https://huggingface.co/Ultralytics/YOLO11/tree/main\n"
                "   Download yolo11n-pose.onnx\n"
                "   Save as: /tmp/yolo11n-pose.onnx\n"
                "Then restart the Python backend.\n")
            raise RuntimeError(error_msg)


def load_yolo_pose_model():
    """Load YOLOv11-pose model with ONNX Runtime"""
    global yolo_pose_net
    if yolo_pose_net is None:
        download_yolo_pose_model()
        yolo_pose_net = ort.InferenceSession(str(YOLO_POSE_MODEL_PATH),
                                             providers=['CPUExecutionProvider'])
        print("✓ YOLOv11-pose model loaded successfully with ONNX Runtime")
    return yolo_pose_net


def detect_objects_yolo(image_array):
    """Detect objects using YOLOv11 with ONNX Runtime"""
    session = load_yolo_model()

    # Prepare image for YOLO (640x640 input)
    img_height, img_width = image_array.shape[:2]
    input_size = 640

    print(f"🔍 Input image shape: {image_array.shape}")

    # Resize and normalize
    input_image = cv2.resize(image_array, (input_size, input_size))
    input_image = input_image.transpose(2, 0, 1)  # HWC to CHW
    input_image = np.expand_dims(input_image, axis=0).astype(
        np.float32) / 255.0

    print(
        f"🔍 Preprocessed image shape: {input_image.shape}, dtype: {input_image.dtype}"
    )
    print(
        f"🔍 Image value range: [{input_image.min():.3f}, {input_image.max():.3f}]"
    )

    # Run inference with ONNX Runtime
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_image})

    # Process outputs (YOLOv11 output format: [1, 84, 8400])
    print(f"🔍 Model output shape: {outputs[0].shape}")
    output = outputs[0][0].transpose()  # [8400, 84]
    print(f"🔍 Transposed output shape: {output.shape}")

    detections = []
    conf_threshold = 0.35  # Confidence threshold for detection
    nms_threshold = 0.50  # IoU threshold for Non-Maximum Suppression

    boxes = []
    confidences = []
    class_ids = []

    # Track top detections for debugging
    top_detections = []

    for detection in output:
        # First 4 values are box coordinates, rest are class scores
        box = detection[:4]
        scores = detection[4:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Track top 5 detections for debugging
        if len(top_detections) < 5:
            top_detections.append({
                'class_id':
                int(class_id),
                'class':
                COCO_CLASSES[class_id]
                if class_id < len(COCO_CLASSES) else f"class_{class_id}",
                'confidence':
                float(confidence),
                'box':
                box.tolist()
            })
        elif confidence > min(d['confidence'] for d in top_detections):
            # Replace lowest confidence
            min_idx = min(range(len(top_detections)),
                          key=lambda i: top_detections[i]['confidence'])
            top_detections[min_idx] = {
                'class_id':
                int(class_id),
                'class':
                COCO_CLASSES[class_id]
                if class_id < len(COCO_CLASSES) else f"class_{class_id}",
                'confidence':
                float(confidence),
                'box':
                box.tolist()
            }

        if confidence > conf_threshold:
            # YOLO format: center_x, center_y, width, height (normalized to input_size)
            center_x, center_y, width, height = box

            # Convert to corner coordinates
            x = center_x - width / 2
            y = center_y - height / 2

            boxes.append([float(x), float(y), float(width), float(height)])
            confidences.append(float(confidence))
            class_ids.append(int(class_id))

    # Log top detections
    print(f"🔍 Top 5 detections (before threshold):")
    for det in sorted(top_detections,
                      key=lambda d: d['confidence'],
                      reverse=True):
        print(f"   {det['class']:15s} - confidence: {det['confidence']:.4f}")

    print(f"🔍 Found {len(boxes)} detections above threshold {conf_threshold}")

    # Apply Non-Maximum Suppression
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                                   nms_threshold)
        print(f"🔍 After NMS: {len(indices)} detections")

        for i in indices:
            box = boxes[i]
            confidence = confidences[i]
            class_id = class_ids[i]

            # Convert to percentages
            x_percent = (box[0] / input_size) * 100
            y_percent = (box[1] / input_size) * 100
            width_percent = (box[2] / input_size) * 100
            height_percent = (box[3] / input_size) * 100

            detections.append({
                "class":
                COCO_CLASSES[class_id]
                if class_id < len(COCO_CLASSES) else f"class_{class_id}",
                "confidence":
                round(confidence, 2),
                "bbox": [
                    round(x_percent, 2),
                    round(y_percent, 2),
                    round(width_percent, 2),
                    round(height_percent, 2)
                ]
            })
            print(
                f"   ✓ Detected: {COCO_CLASSES[class_id]} ({confidence:.2f})")
    else:
        print(f"⚠️  No detections above threshold!")

    print(f"🔍 Returning {len(detections)} final detections")
    return detections


def detect_pose_yolo(image_array):
    """Detect human poses using YOLOv11-pose with ONNX Runtime"""
    session = load_yolo_pose_model()

    # Prepare image for YOLO (640x640 input)
    img_height, img_width = image_array.shape[:2]
    input_size = 640

    print(f"🧍 Input image shape: {image_array.shape}")

    # Resize and normalize
    input_image = cv2.resize(image_array, (input_size, input_size))
    input_image = input_image.transpose(2, 0, 1)  # HWC to CHW
    input_image = np.expand_dims(input_image, axis=0).astype(
        np.float32) / 255.0

    print(
        f"🧍 Preprocessed image shape: {input_image.shape}, dtype: {input_image.dtype}"
    )

    # Run inference with ONNX Runtime
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_image})

    # Process outputs (YOLOv11-pose output format: [1, 56, 8400])
    # 56 = 4 (bbox) + 1 (person conf) + 51 (17 keypoints × 3: x, y, visibility)
    print(f"🧍 Model output shape: {outputs[0].shape}")
    output = outputs[0][0].transpose()  # [8400, 56]
    print(f"🧍 Transposed output shape: {output.shape}")

    detections = []
    conf_threshold = 0.35  # Confidence threshold for person detection
    nms_threshold = 0.50  # IoU threshold for Non-Maximum Suppression

    boxes = []
    confidences = []
    keypoints_list = []

    for detection in output:
        # First 4 values are box coordinates, 5th is person confidence
        box = detection[:4]
        confidence = detection[4]

        if confidence > conf_threshold:
            # YOLO format: center_x, center_y, width, height (normalized to input_size)
            center_x, center_y, width, height = box

            # Convert to corner coordinates
            x = center_x - width / 2
            y = center_y - height / 2

            boxes.append([float(x), float(y), float(width), float(height)])
            confidences.append(float(confidence))

            # Extract keypoints (17 keypoints × 3 values: x, y, visibility)
            kpts = detection[5:].reshape(17, 3)  # [17, 3]
            keypoints_list.append(kpts)

    print(f"🧍 Found {len(boxes)} pose detections above threshold {conf_threshold}")

    # Apply Non-Maximum Suppression
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                                   nms_threshold)
        print(f"🧍 After NMS: {len(indices)} pose detections")

        for i in indices:
            box = boxes[i]
            confidence = confidences[i]
            keypoints = keypoints_list[i]

            # Convert box to percentages
            x_percent = float((box[0] / input_size) * 100)
            y_percent = float((box[1] / input_size) * 100)
            width_percent = float((box[2] / input_size) * 100)
            height_percent = float((box[3] / input_size) * 100)

            # Convert keypoints to percentages and format
            keypoints_formatted = []
            for idx, (kx, ky, kv) in enumerate(keypoints):
                keypoints_formatted.append({
                    "name": COCO_KEYPOINTS[idx],
                    "x": round(float(kx / input_size) * 100, 2),  # percentage
                    "y": round(float(ky / input_size) * 100, 2),  # percentage
                    "visibility": round(float(kv), 2)  # 0-1 confidence
                })

            detections.append({
                "class": "person",
                "confidence": round(float(confidence), 2),
                "bbox": [
                    round(x_percent, 2),
                    round(y_percent, 2),
                    round(width_percent, 2),
                    round(height_percent, 2)
                ],
                "keypoints": keypoints_formatted
            })
            print(f"   ✓ Detected person pose ({confidence:.2f}) with {len(keypoints_formatted)} keypoints")
    else:
        print(f"⚠️  No pose detections above threshold!")

    print(f"🧍 Returning {len(detections)} final pose detections")
    return detections


# Create results directory in project root
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "camera-copilot-ai"}


@app.post("/api/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Detect objects in an image using YOLOv11 with ONNX Runtime
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image_array = np.array(image)

        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Run YOLOv11 detection with ONNX Runtime
        detections = detect_objects_yolo(image_array)

        return {
            "detections": detections,
            "timestamp": datetime.now().isoformat(),
            "model": "YOLOv11n-ONNX"
        }

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Detection failed: {str(e)}")


@app.post("/api/detect-pose")
async def detect_pose(file: UploadFile = File(...)):
    """
    Detect human poses in an image using YOLOv11-pose with ONNX Runtime
    Returns person detections with 17 keypoints (nose, eyes, shoulders, elbows, wrists, hips, knees, ankles)
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image_array = np.array(image)

        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Run YOLOv11-pose detection with ONNX Runtime
        detections = detect_pose_yolo(image_array)

        return {
            "detections": detections,
            "timestamp": datetime.now().isoformat(),
            "model": "YOLOv11n-pose-ONNX"
        }

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Pose detection failed: {str(e)}")


@app.post("/api/detect-stream")
async def detect_objects_stream(
    file: UploadFile = File(...),
    execution_id: str = Form(...),
    target_class: str = Form("boat")
):
    """
    Detect objects with motion tracking across frames.
    Returns detections + motion state (approaching/docking/stationary).
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image_array = np.array(image)

        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Run YOLOv11 detection
        detections = detect_objects_yolo(image_array)
        
        # Get or create execution session
        if execution_id not in execution_sessions:
            execution_sessions[execution_id] = ExecutionSession(execution_id, target_class)
            print(f"🎬 Created new execution session: {execution_id} (tracking: {target_class})")
        
        session = execution_sessions[execution_id]
        
        # Update motion state
        timestamp = time.time()
        motion_state, velocity, should_analyze = session.add_detection(detections, timestamp)
        
        # Buffer frame continuously while stationary (keep buffer fresh)
        if motion_state == "stationary":
            # Convert image to base64 for storage
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            session.buffered_frame = base64.b64encode(buffered.getvalue()).decode('utf-8')
            session.buffered_detections = detections.copy()
            session.buffered_timestamp = timestamp
            if should_analyze:
                print(f"🎞️ [{execution_id}] Buffered initial stationary frame with {len(detections)} detections")
        elif session.buffered_frame:
            # Clear buffer when leaving stationary state
            session.buffered_frame = None
            session.buffered_detections = []
            session.buffered_timestamp = None
            print(f"🗑️ [{execution_id}] Cleared buffer (no longer stationary)")
        
        # Get state summary
        state_summary = session.get_state_summary()
        
        print(f"📊 [{execution_id}] State: {motion_state}, Velocity: {velocity*100:.2f}%/s, Analyze: {should_analyze}")

        return {
            "detections": detections,
            "motion_state": motion_state,
            "velocity": round(velocity * 100, 2),  # % per second
            "should_analyze": should_analyze,
            "state_summary": state_summary,
            "timestamp": datetime.now().isoformat(),
            "model": "YOLOv11n-ONNX"
        }

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Detection failed: {str(e)}")


@app.post("/api/analyze")
async def analyze_detections(data: Dict[str, Any]):
    """
    Analyze detected objects using OpenRouter API with user-defined prompts.
    Supports single images or SEQUENCES of images for temporal action analysis.
    """
    try:
        # Handle both single image and multi-image sequences
        images_base64 = data.get("images", [])
        single_image = data.get("image", "")
        
        # Fallback for older frontend code that only sends one image
        if not images_base64 and single_image:
            images_base64 = [single_image]
            
        detections = data.get("detections", [])
        context = data.get("context", "general")
        user_prompt = data.get("userPrompt", "")
        execution_id = data.get("execution_id")
        is_demo_flow = data.get("isDemoFlow", False)
        
        motion_state = data.get("motion_state")
        velocity = data.get("velocity")
        
        # If using motion tracking and we have a buffered frame, use it instead
        if execution_id and execution_id in execution_sessions:
            session = execution_sessions[execution_id]
            
            if motion_state == "stationary":
                if not session.buffered_frame:
                    return {
                        "analysis": {
                            "condition_met": False,
                            "summary": "Error - no buffered frame available",
                            "extractedData": {},
                            "confidence": 0.0
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                
                print(f"🎞️ [{execution_id}] Using buffered stationary frame for analysis")
                images_base64 = [session.buffered_frame] # Override with buffer
                detections = session.buffered_detections
                
                # Clear buffer after use
                session.buffered_frame = None
                session.buffered_detections = []
                session.buffered_timestamp = None

        # Create analysis prompt
        has_pose_data = any(d.get("keypoints") for d in detections)
        
        if has_pose_data:
            pose_descriptions = []
            for i, detection in enumerate(detections):
                if detection.get("keypoints"):
                    keypoints = detection["keypoints"]
                    visible_kps = [kp for kp in keypoints if kp.get("visibility", 0) > 0.5]
                    kp_data = ", ".join([f"{kp['name']}({kp['x']:.1f}%, {kp['y']:.1f}%)" for kp in visible_kps])
                    pose_descriptions.append(f"Person {i+1} keypoints: {kp_data}")
            detected_objects_str = "; ".join(pose_descriptions)
        else:
            detected_objects_str = ", ".join([d.get("class", "object") for d in detections])
        
        motion_context = ""
        if motion_state and velocity is not None:
            if motion_state == "stationary":
                motion_context = f"\n\nMOTION CONTEXT: The detected object has stopped moving (velocity: {velocity}%/s)."
            elif motion_state == "docking":
                motion_context = f"\n\nMOTION CONTEXT: The detected object is slowing down (velocity: {velocity}%/s)."
            elif motion_state == "approaching":
                motion_context = f"\n\nMOTION CONTEXT: The detected object is moving (velocity: {velocity}%/s)."

        # Sequence guidance based on how many images we have
        sequence_guidance = f"You are analyzing a sequence of {len(images_base64)} frames in chronological order. Look at the progression between them to determine complex actions." if len(images_base64) > 1 else "You are analyzing a single frame."
        
        prompt = f"""You are an elite AI security and behavioral analysis system. {sequence_guidance}

Detected objects in latest frame: {detected_objects_str}{motion_context}

USER REQUIREMENT: {user_prompt}

CRITICAL RULES:
1. Look at the PROGRESSION of the action across the frames.
2. ONLY describe what you can ACTUALLY SEE happening.
3. If no objects are visible, set condition_met=false with summary "No object detected in frame"

Be CONCISE. Analyze the sequence and check if it meets the requirements.
Return a JSON response with:
{{
  "condition_met": true/false,
  "summary": "1-2 sentence summary of the action occurring",
  "extractedData": {{"key": "value pairs if requested"}},
  "confidence": 0.0-1.0
}}

IMPORTANT: Keep summary under 100 characters. Return ONLY valid JSON."""

        if not OPENROUTER_API_KEY:
            raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

        # --- OPENROUTER API INTEGRATION ---
        
        # 1. Start with the text prompt
        content_array = [
            {
                "type": "text",
                "text": prompt
            }
        ]

        # 2. Append all images to the content array
        for img_b64 in images_base64:
            # OpenRouter requires the 'data:image/jpeg;base64,' prefix
            formatted_b64 = img_b64 if img_b64.startswith("data:image") else f"data:image/jpeg;base64,{img_b64}"
            
            content_array.append({
                "type": "image_url",
                "image_url": {
                    "url": formatted_b64
                }
            })

        # 3. Build the payload
        payload = {
            "model": "google/gemini-2.5-flash", # Use OpenRouter's model ID format
            "messages": [
                {
                    "role": "user",
                    "content": content_array
                }
            ]
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000", # Required by OpenRouter
            "X-Title": "SyncroFlow App" # Optional identification
        }

        # 4. Make the request
        print(f"🚀 Sending {len(images_base64)} frames to OpenRouter...")
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if not response.ok:
            raise HTTPException(status_code=500, detail=f"OpenRouter API Error: {response.text}")

        # Parse OpenRouter response
        result_json = response.json()
        text = result_json["choices"][0]["message"]["content"].strip()

        # Clean JSON formatting
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            analysis = json.loads(text)
            if "summary" in analysis and len(analysis["summary"]) > 150:
                analysis["summary"] = analysis["summary"][:147] + "..."
        except:
            analysis = {
                "condition_met": True,
                "summary": text[:100] + ("..." if len(text) > 100 else ""),
                "extractedData": {},
                "confidence": 0.8
            }

        return {"analysis": analysis, "timestamp": datetime.now().isoformat()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe audio using OpenRouter API with speaker diarization support.
    Supports WebM, MP3, WAV, and other common audio formats.
    """
    try:
        print(f"📝 Received audio for transcription: {audio.filename}, type: {audio.content_type}")
        
        # Read audio file
        audio_bytes = await audio.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Determine mime type
        mime_type = audio.content_type or "audio/webm"
        
        print(f"🎤 Transcribing audio ({len(audio_bytes)} bytes, {mime_type})...")
        
        if not OPENROUTER_API_KEY:
            raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

        audio_prompt = """Transcribe this audio with speaker diarization if multiple speakers are detected.
Format the transcript as follows:
- If one speaker: Just provide the plain transcript
- If multiple speakers: Use "Speaker 1:", "Speaker 2:", etc. to indicate different speakers
- Include timestamps in format [MM:SS] for each speaker change
- Be accurate and include filler words (um, uh, etc.) for authenticity
Return ONLY the transcript text, no JSON, no explanations."""

        payload = {
            "model": "google/gemini-2.5-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": audio_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{audio_base64}"}}
                    ]
                }
            ]
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        
        if not response.ok:
            raise HTTPException(status_code=500, detail=f"OpenRouter API error: {response.text}")
            
        transcript = response.json()["choices"][0]["message"]["content"].strip()
        
        print(f"✅ Transcription complete ({len(transcript)} chars)")
        
        return {
            "transcript": transcript,
            "timestamp": datetime.now().isoformat(),
            "audio_duration_bytes": len(audio_bytes)
        }
        
    except Exception as e:
        print(f"❌ Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/api/save-results")
async def save_results(request_data: Dict[str, Any]):
    """
    Save analysis results to file system (JSON or CSV)
    """
    try:
        # Extract data from request
        data = request_data.get("data", {})
        format = request_data.get("format", "json")
        filename = request_data.get("filename")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = filename or f"result_{timestamp}"

        if format == "json":
            filepath = RESULTS_DIR / f"{base_filename}.json"
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

        elif format == "csv":
            filepath = RESULTS_DIR / f"{base_filename}.csv"

            # Flatten data for CSV
            rows = []
            if isinstance(data, dict):
                if "extractedData" in data:
                    rows = [data["extractedData"]]
                else:
                    rows = [data]
            elif isinstance(data, list):
                rows = data
            else:
                rows = [{"data": str(data)}]

            if rows:
                with open(filepath, 'w', newline='') as f:
                    if rows:
                        fieldnames = list(rows[0].keys())
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
        else:
            raise HTTPException(status_code=400,
                                detail="Format must be 'json' or 'csv'")

        return {
            "success": True,
            "filepath": str(filepath),
            "filename": filepath.name,
            "format": format,
            "timestamp": timestamp
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")


@app.get("/api/results")
async def list_results():
    """
    List all saved result files
    """
    try:
        files = []
        for filepath in sorted(RESULTS_DIR.glob("*"),
                               key=lambda p: p.stat().st_mtime,
                               reverse=True):
            if filepath.is_file():
                files.append({
                    "filename":
                    filepath.name,
                    "size":
                    filepath.stat().st_size,
                    "modified":
                    datetime.fromtimestamp(
                        filepath.stat().st_mtime).isoformat(),
                    "path":
                    str(filepath)
                })

        return {"files": files, "count": len(files)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List failed: {str(e)}")


@app.post("/api/generate-flow")
async def generate_flow(data: Dict[str, Any]):
    """
    Generate a workflow configuration from a natural language description using OpenRouter AI
    """
    try:
        prompt_text = data.get("prompt", "")
        if not prompt_text:
            raise HTTPException(status_code=400,
                                detail="Prompt is required")

        # Create prompt for AI to generate flow JSON
        system_prompt = """You are a workflow automation expert. Generate a Camera Copilot flow configuration based on user descriptions.

Available node types:

VISUAL WORKFLOW NODES:
1. camera - Captures video/images from webcam, screen, or uploaded files
   - inputMode: 'webcam' (live webcam feed) | 'screen' (screen recording with optional audio) | 'upload' (pre-recorded video)
   - For screen mode: can capture system audio by enabling "Include Audio" option
   - Processes at 5 FPS for continuous monitoring
   
2. detection - YOLOv11n object detection (80 COCO classes)
   - objectFilter: array of object classes to detect, e.g., ['person'], ['car', 'truck', 'bus'], or [] for all objects
   - Examples: [] = all 80 classes, ['person'] = only people, ['car', 'truck'] = vehicles only
   - Draws bounding boxes with confidence scores
   - Motion tracking: only works when exactly ONE object is selected (e.g., ['boat'] for ferry tracking)
   
3. pose - YOLOv11n-pose human pose detection (17 keypoints)
   - Detects human body keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
   - Perfect for: fitness tracking, sports analysis, safety compliance, gesture recognition
   - Returns skeleton with confidence scores for each keypoint
   - No configuration needed - automatically detects all people in frame
   
4. analysis - AI analysis using Gemini (for VISUAL flows)
   - userPrompt: describe what to analyze in detected objects/scenes
   - Can include motion context (e.g., "is the boat stationary now?")
   - Receives detection metadata and frame analysis

AUDIO WORKFLOW NODES:
4. transcription - Real-time microphone speech-to-text with voice-activated triggers
   - Captures microphone audio continuously
   - Transcribes in real-time using Gemini API
   - VOICE-ACTIVATED: Can detect trigger phrases to auto-stop recording
   - triggerPhrase: spoken phrase that automatically ends recording (e.g., "okay SyncroFlow, meeting ended")
   - Perfect for: meeting notes, voice memos, hands-free automation
   - IMPORTANT: This is the STARTING node for audio-based flows (no camera needed!)
   
5. analysis - AI analysis using Gemini (for AUDIO flows) 
   - userPrompt: describe what to do with transcript (e.g., "Summarize the meeting", "Extract action items")
   - Can use {{transcript}} placeholder in action nodes

ACTION NODES (work with both visual AND audio flows):
6. email - Send email notification
   - to: recipient email, subject: email subject, body: email content
   - Placeholders: {{analysis}}, {{timestamp}}, {{transcript}} (for audio flows)
   
7. sms - Send SMS text message
   - to: phone with country code (e.g., "+1234567890"), message: text content
   - Placeholders: {{analysis}}, {{timestamp}}, {{transcript}} (for audio flows)
   
8. call - Make voice phone call with text-to-speech
   - to: phone with country code, message: spoken message
   - Placeholders: {{analysis}}, {{timestamp}}, {{transcript}} (for audio flows)
   
9. discord - Send Discord webhook message  
   - webhookUrl: Discord webhook URL, message: message content (supports markdown)
   - Placeholders: {{analysis}}, {{timestamp}}, {{transcript}} (for audio flows)

COCO classes (80 total): person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

Flow structure:
{{
  "name": "descriptive name",
  "nodes": [
    {{
      "id": "unique-id",
      "type": "camera|detection|pose|analysis|transcription|email|sms|call|discord",
      "position": {{"x": number, "y": number}},
      "data": {{
        "label": "Node Label",
        "config": {{ /* type-specific config */ }}
      }}
    }}
  ],
  "edges": [
    {{
      "id": "edge-id",
      "source": "source-node-id",
      "target": "target-node-id"
    }}
  ]
}}

EXAMPLE 1 - Visual flow with email notification:
{{
  "name": "Person Detection Email Alert",
  "nodes": [
    {{"id": "camera-1", "type": "camera", "position": {{"x": 100, "y": 100}}, "data": {{"label": "Webcam", "config": {{"inputMode": "webcam"}}}}}},
    {{"id": "detect-1", "type": "detection", "position": {{"x": 400, "y": 100}}, "data": {{"label": "Detect People", "config": {{"objectFilter": ["person"]}}}}}},
    {{"id": "analyze-1", "type": "analysis", "position": {{"x": 700, "y": 100}}, "data": {{"label": "AI Analysis", "config": {{"userPrompt": "Analyze if person detected"}}}}}},
    {{"id": "email-1", "type": "email", "position": {{"x": 1000, "y": 100}}, "data": {{"label": "Send Email", "config": {{"to": "user@example.com", "subject": "Person Detected", "body": "Alert: {{analysis}} at {{timestamp}}"}}}}}}
  ],
  "edges": [
    {{"id": "e1", "source": "camera-1", "target": "detect-1"}},
    {{"id": "e2", "source": "detect-1", "target": "analyze-1"}},
    {{"id": "e3", "source": "analyze-1", "target": "email-1"}}
  ]
}}

EXAMPLE 2 - Voice-activated meeting notes (audio flow):
{{
  "name": "Voice Meeting Notes",
  "nodes": [
    {{"id": "transcription-1", "type": "transcription", "position": {{"x": 100, "y": 100}}, "data": {{"label": "Record Meeting", "config": {{"triggerPhrase": "okay SyncroFlow, meeting ended"}}}}}},
    {{"id": "analyze-1", "type": "analysis", "position": {{"x": 400, "y": 100}}, "data": {{"label": "Summarize", "config": {{"userPrompt": "Summarize the meeting transcript and extract action items"}}}}}},
    {{"id": "email-1", "type": "email", "position": {{"x": 700, "y": 100}}, "data": {{"label": "Email Summary", "config": {{"to": "user@example.com", "subject": "Meeting Summary", "body": "{{analysis}}\n\nFull transcript:\n{{transcript}}"}}}}}}
  ],
  "edges": [
    {{"id": "e1", "source": "transcription-1", "target": "analyze-1"}},
    {{"id": "e2", "source": "analyze-1", "target": "email-1"}}
  ]
}}

EXAMPLE 3 - Fitness pose tracking (visual flow with pose detection):
{{
  "name": "Exercise Form Monitor",
  "nodes": [
    {{"id": "camera-1", "type": "camera", "position": {{"x": 100, "y": 100}}, "data": {{"label": "Webcam", "config": {{"inputMode": "webcam"}}}}}},
    {{"id": "pose-1", "type": "pose", "position": {{"x": 400, "y": 100}}, "data": {{"label": "Detect Pose"}}}},
    {{"id": "analyze-1", "type": "analysis", "position": {{"x": 700, "y": 100}}, "data": {{"label": "Check Form", "config": {{"userPrompt": "Analyze the person's exercise form and posture"}}}}}},
    {{"id": "sms-1", "type": "sms", "position": {{"x": 1000, "y": 100}}, "data": {{"label": "Form Alert", "config": {{"to": "+1234567890", "message": "Form check: {{analysis}}"}}}}}}
  ],
  "edges": [
    {{"id": "e1", "source": "camera-1", "target": "pose-1"}},
    {{"id": "e2", "source": "pose-1", "target": "analyze-1"}},
    {{"id": "e3", "source": "analyze-1", "target": "sms-1"}}
  ]
}}

CRITICAL RULES - YOU MUST FOLLOW THESE:

WORKFLOW TYPE DETECTION:
- If user mentions: "meeting", "transcribe", "voice", "speech", "audio", "recording", "dictation" → Use TRANSCRIPTION flow
- If user mentions: "camera", "webcam", "screen", "detect", "object", "video", "monitor" → Use VISUAL flow
- TRANSCRIPTION flows start with transcription node (NO camera/detection nodes)
- VISUAL flows start with camera node, optionally followed by detection

ACTION NODE RULES:
1. When user says "send email" or "email notification" → ADD a separate "email" type node AFTER analysis
2. When user says "send SMS" or "text message" → ADD a separate "sms" type node AFTER analysis  
3. When user says "call" or "phone alert" → ADD a separate "call" type node AFTER analysis
4. When user says "discord" or "discord message" → ADD a separate "discord" type node AFTER analysis
5. The analysis node does the AI thinking, action nodes (email/sms/call/discord) do the sending
6. NEVER put notification logic in the analysis node userPrompt - always create a separate action node

POSITIONING:
- Visual flows: camera(100,100), detection(400,100), analysis(700,100), action(1000,100)
- Audio flows: transcription(100,100), analysis(400,100), action(700,100)
- Use 300px horizontal spacing between nodes

Node configurations (REQUIRED examples):
- transcription: {{"triggerPhrase": "okay SyncroFlow, meeting ended"}} OR {{"triggerPhrase": "stop recording"}}
- camera: {{"inputMode": "webcam"}} OR {{"inputMode": "screen"}} OR {{"inputMode": "upload"}}
- detection: {{"objectFilter": ["person"]}} OR {{"objectFilter": ["car", "truck"]}} OR {{"objectFilter": []}} for all objects
- pose: no configuration needed (automatically detects all people)
- analysis (visual): {{"userPrompt": "Analyze detected objects and their behavior"}}
- analysis (audio): {{"userPrompt": "Summarize the meeting and extract action items"}}
- email: {{"to": "user@example.com", "subject": "Alert", "body": "{{analysis}} at {{timestamp}}"}}
- sms: {{"to": "+1234567890", "message": "Alert: {{analysis}} at {{timestamp}}"}}
- call: {{"to": "+1234567890", "message": "Alert: {{analysis}}"}}
- discord: {{"webhookUrl": "[https://discord.com/api/webhooks/123/abc](https://discord.com/api/webhooks/123/abc)", "message": "Alert: {{analysis}} at {{timestamp}}"}}

IMPORTANT: objectFilter is ALWAYS an array. Examples:
- "detect people" → {{"objectFilter": ["person"]}}
- "detect cars and trucks" → {{"objectFilter": ["car", "truck"]}}
- "detect all objects" → {{"objectFilter": []}}
- "monitor for boats" → {{"objectFilter": ["boat"]}}
ONLY return valid JSON, no explanations

USER REQUEST: {user_prompt}

Generate the flow JSON:"""

        if not OPENROUTER_API_KEY:
            raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

        payload = {
            "model": "google/gemini-2.5-flash",
            "messages": [
                {"role": "user", "content": system_prompt.format(user_prompt=prompt_text)}
            ]
        }
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        
        if not response.ok:
            raise HTTPException(status_code=500, detail=f"OpenRouter API error: {response.text}")

        # Parse response - handle various formats AI might return
        try:
            text = response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[GENERATE-FLOW] Error accessing response.text: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to get response text: {str(e)}")
        
        print(f"[GENERATE-FLOW] Raw response (first 500 chars): {text[:500]}")
        
        # Remove markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        # Find JSON object - look for opening brace
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        print(f"[GENERATE-FLOW] JSON bounds: start={start_idx}, end={end_idx}")
        
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx + 1]
        
        print(f"[GENERATE-FLOW] Extracted JSON (first 300 chars): {text[:300]}")

        try:
            flow_config = json.loads(text)
            print(f"[GENERATE-FLOW] Successfully parsed JSON with keys: {flow_config.keys()}")

            # Validate structure
            if "nodes" not in flow_config or "edges" not in flow_config:
                raise ValueError("Invalid flow structure")

            # Ensure name exists
            if "name" not in flow_config:
                flow_config["name"] = "Generated Flow"

            # Post-processing: Add missing action nodes if user requested notifications
            prompt_lower = prompt_text.lower()
            node_types = [node.get("type") for node in flow_config.get("nodes", [])]
            
            # Check if we need to add action nodes
            needs_email = ("email" in prompt_lower or "mail" in prompt_lower) and "email" not in node_types
            needs_sms = ("sms" in prompt_lower or "text message" in prompt_lower) and "sms" not in node_types
            needs_call = ("call" in prompt_lower or "phone" in prompt_lower) and "call" not in node_types
            needs_discord = ("discord" in prompt_lower) and "discord" not in node_types
            
            print(f"[GENERATE-FLOW] Post-processing: needs_email={needs_email}, needs_sms={needs_sms}, needs_call={needs_call}, needs_discord={needs_discord}")
            print(f"[GENERATE-FLOW] Existing node types: {node_types}")
            
            # Find the analysis node (to connect action nodes after it)
            analysis_node = next((n for n in flow_config["nodes"] if n["type"] == "analysis"), None)
            print(f"[GENERATE-FLOW] Analysis node found: {analysis_node is not None}")
            
            if analysis_node and (needs_email or needs_sms or needs_call or needs_discord):
                max_x = max(n["position"]["x"] for n in flow_config["nodes"]) + 300
                node_y = analysis_node["position"]["y"]
                
                if needs_email:
                    email_id = f"email-{len(flow_config['nodes']) + 1}"
                    flow_config["nodes"].append({
                        "id": email_id,
                        "type": "email",
                        "position": {"x": max_x, "y": node_y},
                        "data": {
                            "label": "Send Email",
                            "config": {
                                "to": "user@example.com",
                                "subject": "Alert Notification",
                                "body": "{{analysis}} at {{timestamp}}"
                            }
                        }
                    })
                    flow_config["edges"].append({
                        "id": f"edge-{len(flow_config['edges']) + 1}",
                        "source": analysis_node["id"],
                        "target": email_id
                    })
                    print(f"[GENERATE-FLOW] Auto-added email node")
                
                if needs_sms:
                    sms_id = f"sms-{len(flow_config['nodes']) + 1}"
                    flow_config["nodes"].append({
                        "id": sms_id,
                        "type": "sms",
                        "position": {"x": max_x, "y": node_y + 150},
                        "data": {
                            "label": "Send SMS",
                            "config": {
                                "to": "+1234567890",
                                "message": "Alert: {{analysis}} at {{timestamp}}"
                            }
                        }
                    })
                    flow_config["edges"].append({
                        "id": f"edge-{len(flow_config['edges']) + 1}",
                        "source": analysis_node["id"],
                        "target": sms_id
                    })
                    print(f"[GENERATE-FLOW] Auto-added SMS node")
                
                if needs_call:
                    call_id = f"call-{len(flow_config['nodes']) + 1}"
                    flow_config["nodes"].append({
                        "id": call_id,
                        "type": "call",
                        "position": {"x": max_x, "y": node_y + 300},
                        "data": {
                            "label": "Make Call",
                            "config": {
                                "to": "+1234567890",
                                "message": "Alert: {{analysis}}"
                            }
                        }
                    })
                    flow_config["edges"].append({
                        "id": f"edge-{len(flow_config['edges']) + 1}",
                        "source": analysis_node["id"],
                        "target": call_id
                    })
                    print(f"[GENERATE-FLOW] Auto-added call node")
                
                if needs_discord:
                    discord_id = f"discord-{len(flow_config['nodes']) + 1}"
                    flow_config["nodes"].append({
                        "id": discord_id,
                        "type": "discord",
                        "position": {"x": max_x, "y": node_y + 450},
                        "data": {
                            "label": "Send Discord",
                            "config": {
                                "webhookUrl": "https://discord.com/api/webhooks/123456/abcdefg",
                                "message": "Alert: {{analysis}} at {{timestamp}}"
                            }
                        }
                    })
                    flow_config["edges"].append({
                        "id": f"edge-{len(flow_config['edges']) + 1}",
                        "source": analysis_node["id"],
                        "target": discord_id
                    })
                    print(f"[GENERATE-FLOW] Auto-added discord node")

            return {"flow": flow_config, "success": True}

        except (json.JSONDecodeError, ValueError) as e:
            print(f"[GENERATE-FLOW] Parse error: {str(e)}")
            print(f"[GENERATE-FLOW] Raw text: {text[:200]}")
            # Fallback: create basic flow
            flow_config = {
                "name":
                "Generated Flow",
                "nodes": [{
                    "id": "camera-1",
                    "type": "camera",
                    "position": {
                        "x": 100,
                        "y": 100
                    },
                    "data": {
                        "label": "Video Source",
                        "config": {
                            "inputMode": "webcam"
                        }
                    }
                }, {
                    "id": "detect-1",
                    "type": "detection",
                    "position": {
                        "x": 400,
                        "y": 100
                    },
                    "data": {
                        "label": "Object Detection",
                        "config": {
                            "objectFilter": "all"
                        }
                    }
                }],
                "edges": [{
                    "id": "edge-1",
                    "source": "camera-1",
                    "target": "detect-1"
                }]
            }
            return {"flow": flow_config, "success": True}

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Flow generation failed: {str(e)}")


@app.post("/api/edit-flow")
async def edit_flow(data: Dict[str, Any]):
    """
    Edit an existing workflow configuration based on user's natural language request using OpenRouter AI
    """
    try:
        prompt_text = data.get("prompt", "")
        current_flow = data.get("currentFlow", {})
        
        if not prompt_text:
            raise HTTPException(status_code=400, detail="Prompt is required")
        if not current_flow or "nodes" not in current_flow or "edges" not in current_flow:
            raise HTTPException(status_code=400, detail="Valid current flow is required")

        # Create prompt for AI to edit the flow
        system_prompt = """You are a workflow automation expert. Edit an existing SyncroFlow flow configuration based on the user's request.

CURRENT FLOW:
{current_flow_json}

USER REQUEST: {user_request}

Available node types (same as flow generation):
- camera (webcam/screen/upload), detection (YOLO objects), pose (human pose), analysis (Gemini AI)
- transcription (microphone audio), email, sms, call, discord

YOUR TASK:
Modify the above flow according to the user's request. You can:
1. ADD new nodes (e.g., "add email notification")
2. REMOVE nodes (e.g., "remove the SMS alert")
3. MODIFY node configurations (e.g., "change detection to only track cars")
4. CHANGE connections/edges (e.g., "connect pose to analysis")
5. UPDATE node positions for better layout

RULES:
- Maintain all node IDs that aren't being removed
- Use incremental IDs for new nodes (e.g., email-2, sms-3)
- Position new nodes logically (300px spacing)
- Preserve existing configurations unless specifically asked to change
- If adding action nodes, connect them AFTER the analysis node
- For detection changes: use array format like {{"objectFilter": ["person", "car"]}}
- Keep flow type consistent (visual vs audio)

Return ONLY the complete modified flow JSON with "name", "nodes", and "edges". No explanations.

Example modifications:
- "add email" → Insert email node after analysis, connect with edge
- "remove sms" → Delete sms node and its edges
- "detect only people" → Update detection config: {{"objectFilter": ["person"]}}
- "change prompt" → Update analysis userPrompt

Generate the modified flow JSON:"""

        current_flow_json = json.dumps(current_flow, indent=2)
        final_prompt = system_prompt.format(
            current_flow_json=current_flow_json,
            user_request=prompt_text
        )

        if not OPENROUTER_API_KEY:
            raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

        payload = {
            "model": "google/gemini-2.5-flash",
            "messages": [
                {"role": "user", "content": final_prompt}
            ]
        }
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        
        if not response.ok:
            raise HTTPException(status_code=500, detail=f"OpenRouter API error: {response.text}")

        # Parse response
        try:
            text = response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[EDIT-FLOW] Error accessing response.text: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to get response text: {str(e)}")
        
        print(f"[EDIT-FLOW] Raw response (first 500 chars): {text[:500]}")
        
        # Remove markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        # Find JSON object
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx + 1]
        
        print(f"[EDIT-FLOW] Extracted JSON (first 300 chars): {text[:300]}")

        try:
            flow_config = json.loads(text)
            print(f"[EDIT-FLOW] Successfully parsed JSON with keys: {flow_config.keys()}")

            # Validate structure
            if "nodes" not in flow_config or "edges" not in flow_config:
                raise ValueError("Invalid flow structure")

            # Ensure name exists
            if "name" not in flow_config:
                flow_config["name"] = current_flow.get("name", "Edited Flow")

            return {"flow": flow_config, "success": True}

        except (json.JSONDecodeError, ValueError) as e:
            print(f"[EDIT-FLOW] Parse error: {str(e)}")
            print(f"[EDIT-FLOW] Failed text: {text[:500]}")
            # Return original flow if edit fails
            return {"flow": current_flow, "success": False, "error": "Failed to parse AI response"}

    except Exception as e:
        print(f"[EDIT-FLOW] Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Flow edit failed: {str(e)}")


@app.post("/api/execution-session/clear")
async def clear_execution_session(data: Dict[str, Any]):
    """
    Clear an execution session to free up memory.
    """
    try:
        execution_id = data.get("execution_id")
        if not execution_id:
            raise HTTPException(status_code=400, detail="execution_id required")
        
        if execution_id in execution_sessions:
            del execution_sessions[execution_id]
            print(f"🗑️ Cleared execution session: {execution_id}")
            return {"success": True, "message": f"Session {execution_id} cleared"}
        else:
            return {"success": True, "message": f"Session {execution_id} not found (already cleared)"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear session failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PYTHON_PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)