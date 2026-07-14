import os
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n-seg.pt") 
    current_dir = os.path.dirname(os.path.abspath(__file__))

    model.train(
        data="yolo_dataset/dataset.yaml", 
        epochs=50,         
        imgsz=640,         
        batch=16,          
        device='cpu',  
        project=current_dir, 
        name="YOLO_My_Model_1" 
    )