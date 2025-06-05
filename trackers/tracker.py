from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import pandas as pd
import pickle
import os
import sys
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

# Add the parent directory to the system path to import utils
sys.path.append("../")


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        # For adding positions to tracks object
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object == "ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position 
    
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]  # Empty list will be interpolated
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()   # Backward fill to handle remaining NaNs if initial frames are empty

        # Convert DataFrame back to list of dictionaries of lists
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        # Batch size to prevent memory overflow
        batch_size = 20
        detections = []
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        # If read_from_stub is True, load 'tracks' object from stub_path
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks 

        # Detect objects in frames
        detections = self.detect_frames(frames)

        # Initialize tracks dictionary
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            # Inverse mapping of class names (eg: "0:person" to "person:0")
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}
            print(cls_names)

            # Convert detections to Supervision Detection Format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]
            
            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # Initialize empty tracks for each frame
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Iterate through detections with tracks
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # Need to track only one ball, so no need for track_id
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # Dump tracks to stub_path if provided
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        # Draw an ellipse on the frame based on the bounding box
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Draw the ellipse
        cv2.ellipse(
            frame,
            center = (x_center, y2),
            axes = (int(width), int(0.35*width)),
            angle = 0.0,
            startAngle = -45,
            endAngle = 235,
            color = color,
            thickness = 2,
            lineType = cv2.LINE_4
        )

        # Draw the rectangle below the ellipse
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15  # 15 is buffer for better visibility
        y2_rect = (y2 + rectangle_height//2) + 15  # 15 is buffer for better visibility

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
            )

            # For better visuals (padding for text)
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, # size
                (0,0,0), # text color - black
                2 # thickness
            )
        
        return frame
    
    def draw_triangle(self, frame, bbox, color):
        # For ball pointer
        y = int(bbox[1]) # y-coordinate of the top-left corner
        x, _ = get_center_of_bbox(bbox) # x-coordinate of the center

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])
        
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)  # Draw black border

        return frame
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (1390, 0),
            (1920, 95),
            (255,255,255),
            cv2.FILLED
        )
        alpha = 0.8
        cv2.addWeighted(
            overlay,
            alpha,
            frame,
            1 - alpha,
            0,
            frame
        )

        # Get the number of times each team has the ball
        team_ball_control_till_frame = team_ball_control[:frame_num + 1] # Get team ball control till current frame
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        # Draw the text on the frame
        cv2.putText(
            frame,
            f"Team 1's Ball Possession: {team_1*100:.2f}%",
            (1405, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,0,0),
            3
        )
        cv2.putText(
            frame,
            f"Team 2's Ball Possession: {team_2*100:.2f}%",
            (1405, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,0,0),
            3
        )

        return frame
    
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        # Draw annotations on video frames based on the tracks
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (0,0,255))  # red triangle for player with ball

            # Draw referees (not tracking them, so no track_id)
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0,255,255))  # yellow ellipse for referee

            # Draw ball
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))  # green triangle for ball

            # Draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)    

            output_video_frames.append(frame)
        
        return output_video_frames