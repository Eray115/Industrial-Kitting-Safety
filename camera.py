import os
import cv2
import numpy as np
from pypylon import pylon
from ultralytics import YOLO
import datetime

# Directory and model paths
current_dir = os.path.dirname(os.path.abspath(__file__))
# weight file 
model_path = os.path.join(current_dir, "YOLO_My_Model_1", "weights", "best.pt")

if os.path.exists(model_path):
    model = YOLO(model_path)
    print(f"Successfully loaded model from: {model_path}")
else:
    print(f"ERROR: Model not found at {model_path}")
    exit()

# Initialize the Basler camera using pypylon
try:
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    print("Successfully connected to Basler Camera.")
except Exception as e:
    print(f"CAMERA ERROR: Could not connect to Basler camera. Details: {e}")
    exit()

# format converter for OpenCV compatibility (BGR format)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

cv2.namedWindow('Live Industrial Zone Analysis', cv2.WINDOW_NORMAL)

# RECTANGULAR ZONES 
WARNING_ZONE = (620, 120, 1910, 1050)  
DANGER_ZONE = (850, 340, 1720, 950)    
KITTING_ZONE = (1040, 540, 1490, 800)  

def is_point_inside_rect(point, rect):
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2

print("\n--- Live Kitting & Safety Analysis Started (Press 'q' to quit) ---")

while camera.IsGrabbing():
    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    
    if grab_result.GrabSucceeded():
        # Convert Basler camera frame to OpenCV image array
        image = converter.Convert(grab_result)
        img = image.GetArray()
        
        # Real-time clock stamp for live logs
        current_timestamp = datetime.datetime.now()
        timestamp_str = current_timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # Execute inference on live hardware frame
        results = model.predict(img, conf=0.25, iou=0.45, classes=[0, 1, 2, 3], verbose=False)
        vis_frame = results[0].plot()
        
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

        # LIVE RECTANGULAR SAFETY 
        safety_state = "SAFETY_OK"
        ui_color = (0, 255, 0)

        if len(hand_centers) > 0:
            for h_center in hand_centers:
                if is_point_inside_rect(h_center, DANGER_ZONE):
                    safety_state = "DANGER"
                    ui_color = (0, 0, 255)
                    break
                elif is_point_inside_rect(h_center, WARNING_ZONE):
                    safety_state = "WARNING"
                    ui_color = (0, 165, 255)

        # LIVE PARALLEL KITTING FSM LOGIC
        if red_inside_kit and green_inside_kit:
            kit_state = "KIT_COMPLETED"
            kit_color = (0, 255, 0)
        elif red_inside_kit or green_inside_kit:
            kit_state = "KIT_PARTIAL"
            kit_color = (0, 255, 255)
        else:
            kit_state = "KIT_EMPTY"
            kit_color = (255, 255, 255)

        #DRAW STATIC ZONE RECTANGLES ON LIVE FRAME
        cv2.rectangle(vis_frame, (WARNING_ZONE[0], WARNING_ZONE[1]), (WARNING_ZONE[2], WARNING_ZONE[3]), (0, 165, 255), 3)
        cv2.putText(vis_frame, "WARNING ZONE", (WARNING_ZONE[0] + 20, WARNING_ZONE[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        cv2.rectangle(vis_frame, (DANGER_ZONE[0], DANGER_ZONE[1]), (DANGER_ZONE[2], DANGER_ZONE[3]), (0, 0, 255), 3)
        cv2.putText(vis_frame, "DANGER ZONE", (DANGER_ZONE[0] + 20, DANGER_ZONE[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.rectangle(vis_frame, (KITTING_ZONE[0], KITTING_ZONE[1]), (KITTING_ZONE[2], KITTING_ZONE[3]), (255, 128, 0), 3)
        cv2.putText(vis_frame, "KITTING TRAY", (KITTING_ZONE[0] + 20, KITTING_ZONE[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)

        #INDUSTRIAL HUD PANEL EFFECT
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (30, 30), (620, 230), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)
        cv2.rectangle(vis_frame, (30, 30), (620, 230), ui_color, 3)

        # ONSCREEN OVERLAY
        cv2.putText(vis_frame, f"SAFETY: {safety_state}", (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.3, ui_color, 3)
        cv2.putText(vis_frame, f"KITTING: {kit_state}", (50, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.3, kit_color, 3)
        cv2.putText(vis_frame, f"TIME: {timestamp_str}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        print(f"[{timestamp_str}] LIVE STATE -> Safety: {safety_state} | Kitting: {kit_state}")
        # Exit the progeram
        cv2.imshow("Live Industrial Zone Analysis", vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    grab_result.Release()

# Cleanup hardware connections cleanly
camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
print("Live system shutdown cleanly.")