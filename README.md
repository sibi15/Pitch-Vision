# TactiTrack
A deep learning pipeline that performs end-to-end football video analysis using object detection, tracking, and tactical annotation. It detects players, referees, and the ball using YOLOv8 and Roboflow trained models, tracks their movement, assigns team identities based on jersey color, and computes metrics like ball possession, speed, and distance. The system also handles camera movement and applies perspective transformation for more accurate spatial analysis.

Key Features:
- YOLO-based player and ball detection
- Custom tracking with ellipse/ID overlays
- Jersey color-based team classification
- Ball possession and control metrics
- Perspective correction and camera stabilization
- Speed and distance estimation

Future Work
- Integrate event detection (goals, fouls, passes)
- Add top-down tactical map visualization
- Export annotated video clips for coaching workflows
- Extend team classification beyond jersey color via clustering embeddings

#### Credits
This project was inspired by and built upon the work of Abdullah Tarek – https://github.com/abdullahtarek

All credit goes to the original author for the foundational work. This repository is a modified and extended version for learning and research purposes.
