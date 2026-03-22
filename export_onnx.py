"""
Export best.pt to ONNX format.
Run this on the GCP VM (PyTorch 2.7 + ultralytics 8.1.0 environment).

Usage:
    python export_onnx.py
"""
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(format="onnx", imgsz=640, opset=17)
print("Done — best.onnx created.")
