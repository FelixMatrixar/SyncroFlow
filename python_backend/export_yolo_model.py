#!/usr/bin/env python3
"""
One-time script to export YOLOv11n to ONNX format using Ultralytics.
Run this once to generate the ONNX model, then use it in the main application.

Usage:
    pip install ultralytics
    python export_yolo_model.py
"""

from ultralytics import YOLO
from pathlib import Path

def export_yolo_to_onnx():
    """Export YOLOv11n to ONNX format"""
    print("📥 Downloading YOLOv11n from Ultralytics...")
    model = YOLO("yolo11n.pt")
    
    print("🔄 Exporting to ONNX format...")
    model.export(format="onnx", dynamic=True, simplify=True)
    
    # Move to the expected location
    source = Path("yolo11n.onnx")
    target = Path("/tmp/yolo11n.onnx")
    
    if source.exists():
        source.rename(target)
        print(f"✓ Model exported successfully to {target}")
        print(f"   Size: {target.stat().st_size / 1024 / 1024:.1f}MB")
    else:
        print("❌ Export failed - file not found")

if __name__ == "__main__":
    export_yolo_to_onnx()
