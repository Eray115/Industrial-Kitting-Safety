import cv2
from ultralytics import YOLO
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.normpath(os.path.join(current_dir, "yolov8n-seg.pt"))

print("="*50)
print(f"Model yükleniyor: {model_path}")
print("="*50)

model = YOLO(model_path)

# Kamerayı başlat
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("HATA: Kamera açılmadı!")
    exit()

print("\nKamera açıldı! Sadece senin objelerin filtreleniyor...")
print("Kapatmak için 'q' tuşuna basın.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # CRITICAL FILTERS: 
    # conf=0.25 -> Güven eşiği (%25 ve üzeri tahminleri yakala)
    # classes=[0, 1, 2, 3] -> Sadece senin dataset.yaml dosandaki ilk 4 sınıfı gösterir!
    # Bu ayar sayesinde elma, donut, araba gibi yabancı hiçbir sınıf ekrana çizilmez.
    results = model(frame, conf=0.25, iou=0.45, classes=[0, 1, 2, 3])

    # Sadece filtrelenmiş sınıfların maskelerini ve kutularını çizdiriyoruz
    annotated_frame = results[0].plot()
    
    cv2.imshow("YOLOv8 Sadece Benim Objelerim", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Test bitti.")