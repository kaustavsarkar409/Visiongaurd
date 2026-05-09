from ultralytics import YOLO
import cv2
import torch

print("YOLO Installed")
print("OpenCV Installed")
print("MPS Available:", torch.backends.mps.is_available())
