# Vision Guard

## AI-Powered Smart Campus Surveillance & Student Safety System

Vision Guard is an AI-based smart surveillance and monitoring prototype designed to improve student safety inside universities, colleges, and campus environments.

The system uses Computer Vision, YOLOv8, and OpenCV to monitor campus camera feeds and detect:

- Aggressive behavior
- Ragging-related activities
- Suspicious movements
- Crowd overcrowding
- Restricted-area activity
- Entry and exit movement
- Night-time malicious activity near hostel zones

The project aims to create a safer, smarter, and more responsive campus ecosystem through intelligent real-time surveillance and zone-wise monitoring.

------------------

## Problem Statement

Educational institutions often face challenges related to:

- Ragging and bullying
- Unsafe campus zones
- Delayed response to suspicious activities
- Crowd management issues
- Night-time security concerns
- Lack of intelligent surveillance systems

Traditional CCTV systems only record footage and require manual monitoring, making it difficult to identify incidents proactively.

Vision Guard addresses this problem using AI-powered automated surveillance and anomaly detection.

------------------

## Key Features

## Smart Surveillance

- Real-time camera monitoring
- Zone-wise surveillance system
- Multi-camera support

## AI-Based Detection

- Aggressive behavior detection
- Ragging/suspicious activity detection
- Human presence detection
- Movement tracking

## Crowd Density Monitoring

- Overcrowding detection
- Staircase and corridor congestion alerts
- Ground/common area density analysis

## Hostel Security Monitoring

- Night-time suspicious activity detection
- Unauthorized movement alerts
- Entry/Exit monitoring

## Web Dashboard

- Simple web-based monitoring interface
- Live detection display
- Alert visualization
- Zone monitoring panel

------------------

## Tech Stack

| Component | Technology |
|---|---|
| AI Detection Model | YOLOv8 |
| Computer Vision | OpenCV |
| Programming Language | Python |
| Frontend | HTML, CSS |
| Backend Logic | Python |
| Camera Input | CCTV / Webcam Feed |
| Detection Pipeline | Real-Time Video Processing |

------------------

## Author

- [kaustavsarkar409](https://github.com/kaustavsarkar409/Visiongaurd/edit/main/README.md)
- [souparnodas244-cmd](https://github.com/souparnodas244-cmd)
- [sohambiswas415-hue](https://github.com/sohambiswas415-hue/vision_gaurd)

## System Architecture

```text
                    +----------------------+
                    |   CCTV / Camera Feed |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |   OpenCV Processing  |
                    |  Frame Extraction    |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |   YOLOv8 AI Model    |
                    | Object & Activity    |
                    | Detection Engine     |
                    +----------+-----------+
                               |
              +----------------+----------------+
              |                                 |
              v                                 v

   +--------------------+          +----------------------+
   | Crowd Density      |          | Suspicious Activity  |
   | Analysis Module    |          | Detection Module     |
   +--------------------+          +----------------------+
              |                                 |
              +----------------+----------------+
                               |
                               v
                    +----------------------+
                    | Alert & Monitoring   |
                    | Decision System      |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Website Dashboard    |
                    | HTML + CSS Interface |
                    +----------------------+
        


