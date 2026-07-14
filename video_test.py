import os
import cv2
import numpy as np
from ultralytics import YOLO
import datetime

# Directory and model paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "YOLO_My_Model_1", "weights", "best.pt")
video_path = r"C:\Users\eraya\OneDrive\Desktop\Project\video_1.mp4"

if os.path.exists(model_path):
    model = YOLO(model_path)
    print(f"Successfully loaded model from: {model_path}")
else:
    print(f"ERROR: Model not found at {model_path}")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"ERROR: Could not open video file: {video_path}")
    exit()

cv2.namedWindow('D-Grade Rectangular Zone Demo', cv2.WINDOW_NORMAL)
base_time = datetime.datetime(2026, 5, 30, 15, 3, 34, 423000)

frame_counter = 0
#Process every 4th frame to play the video much faster
FRAME_SKIP_INTERVAL = 4 


WARNING_ZONE = (550, 180, 1900, 1100)  
DANGER_ZONE = (780, 400, 1680, 1000)   
KITTING_ZONE = (980, 600, 1440, 840)   

def is_point_inside_rect(point, rect):
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2

print("\n--- Starting High-Speed Right-Up Shifted Zone Processing (Press 'q' to quit) ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    msec_elapsed = cap.get(cv2.CAP_PROP_POS_MSEC)
    current_timestamp = base_time + datetime.timedelta(milliseconds=msec_elapsed)
    timestamp_str = current_timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]

    if frame_counter % FRAME_SKIP_INTERVAL != 0:
        continue

    # Execute CPU inference
    results = model.predict(frame, conf=0.25, iou=0.45, classes=[0, 1, 2, 3], device='cpu', verbose=False)
    vis_frame = results[0].plot()
    
    detected_classes = []
    hand_centers = []
    red_inside_kit = False
    green_inside_kit = False
    
    if len(results[0].boxes) > 0:
        detected_classes = results[0].boxes.cls.cpu().numpy().astype(int)
        boxes_coords = results[0].boxes.xyxy.cpu().numpy()
        
        for cls_id, box in zip(detected_classes, boxes_coords):
            center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
            
            if cls_id == 3:    # hand
                hand_centers.append(center)
            elif cls_id == 1:  # red_obj
                if is_point_inside_rect(center, KITTING_ZONE):
                    red_inside_kit = True
            elif cls_id == 2:  # green_obj
                if is_point_inside_rect(center, KITTING_ZONE):
                    green_inside_kit = True


    safety_state = "SAFETY_OK"
    ui_color = (0, 255, 0) # Green for OK

    if len(hand_centers) > 0:
        for h_center in hand_centers:
            if is_point_inside_rect(h_center, DANGER_ZONE):
                safety_state = "DANGER"
                ui_color = (0, 0, 255) # Red for Danger
                break
            elif is_point_inside_rect(h_center, WARNING_ZONE):
                safety_state = "WARNING"
                ui_color = (0, 165, 255) # Orange for Warning

 
    if red_inside_kit and green_inside_kit:
        kit_state = "KIT_COMPLETED"
        kit_color = (0, 255, 0)
    elif red_inside_kit or green_inside_kit:
        kit_state = "KIT_PARTIAL"
        kit_color = (0, 255, 255)
    else:
        kit_state = "KIT_EMPTY"
        kit_color = (255, 255, 255)

  
    # Warning Zone Box 
    cv2.rectangle(vis_frame, (WARNING_ZONE[0], WARNING_ZONE[1]), (WARNING_ZONE[2], WARNING_ZONE[3]), (0, 165, 255), 3)
    cv2.putText(vis_frame, "WARNING ZONE", (WARNING_ZONE[0] + 20, WARNING_ZONE[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    # Danger Zone Box 
    cv2.rectangle(vis_frame, (DANGER_ZONE[0], DANGER_ZONE[1]), (DANGER_ZONE[2], DANGER_ZONE[3]), (0, 0, 255), 3)
    cv2.putText(vis_frame, "DANGER ZONE", (DANGER_ZONE[0] + 20, DANGER_ZONE[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Kitting Tray Box 
    cv2.rectangle(vis_frame, (KITTING_ZONE[0], KITTING_ZONE[1]), (KITTING_ZONE[2], KITTING_ZONE[3]), (255, 128, 0), 3)
    cv2.putText(vis_frame, "KITTING TRAY", (KITTING_ZONE[0] + 20, KITTING_ZONE[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)

    
    overlay = vis_frame.copy()
    cv2.rectangle(overlay, (30, 30), (620, 230), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)
    cv2.rectangle(vis_frame, (30, 30), (620, 230), ui_color, 3)

  
    cv2.putText(vis_frame, f"SAFETY: {safety_state}", (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.3, ui_color, 3)
    cv2.putText(vis_frame, f"KITTING: {kit_state}", (50, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.3, kit_color, 3)
    cv2.putText(vis_frame, f"TIME: {timestamp_str}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

 
    print(f"[{timestamp_str}] EMITTED STATE -> Safety: {safety_state} | Kitting: {kit_state}")

  
    cv2.imshow('D-Grade Rectangular Zone Demo', vis_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nVideo processing finished cleanly.")