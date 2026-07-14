import os
import json
import glob
import shutil
import random

current_dir = os.path.dirname(os.path.abspath(__file__))

source_dir = os.path.join(current_dir, "data_pool")
dataset_dir = os.path.join(current_dir, "yolo_dataset")

# Labelme to YOLO Class ID Mapping
class_map = {
    "robot": 0, 
    "red_obj": 1, 
    "green_obj": 2, 
    "hand": 3
}

print("1. Creating dataset directory structure...")
for split in ['train', 'val']:
    os.makedirs(os.path.join(dataset_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, split, 'labels'), exist_ok=True)

print(f"2. Scanning source directory: {source_dir}")

all_files = os.listdir(source_dir)
print(f"--- Total raw files found in source directory: {len(all_files)} ---")

json_files = []
for f in all_files:
    f_lower = f.lower()
    if f_lower.endswith('.json') or '.json.' in f_lower or f_lower.endswith('.json.txt'):
        json_files.append(os.path.join(source_dir, f))

print(f"3. Filtered and validated JSON files count: {len(json_files)}")

if len(json_files) == 0:
    print("\n!!! PROCESS TERMINATED !!!")
    print("Python accessed the folder, but the actual file extensions do not match standard '.json'.")
    print("First 5 files in the directory for structural reference:")
    for item in all_files[:5]:
        print(f" -> {item}")
    input("\nPress Enter to exit and inspect the file extensions manually...")
    exit()

# 85% Train, 15% Validation dataset shuffel
random.shuffle(json_files)
split_idx = int(len(json_files) * 0.85)
train_files = json_files[:split_idx]
val_files = json_files[split_idx:]

def convert_and_distribute(file_list, split):
    print(f"\nProcessing {split.upper()} subset ({len(file_list)} files)...")
    for json_path in file_list:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            img_name = data.get('imagePath', '')
            img_w = data.get('imageWidth', 2448)
            img_h = data.get('imageHeight', 2048)
            
            # Extract base file name
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            # Clean up double extensions
            base_name = base_name.replace('.json', '')
            
            possible_img_names = [img_name, f"{base_name}.jpg", f"{base_name}.png", f"{base_name}.BMP"]
            src_img_path = None
            final_img_name = f"{base_name}.jpg"
            
            for p_name in possible_img_names:
                if p_name:
                    p_path = os.path.join(source_dir, p_name)
                    if os.path.exists(p_path):
                        src_img_path = p_path
                        final_img_name = p_name
                        break
            
            if src_img_path and os.path.exists(src_img_path):
                shutil.copy(src_img_path, os.path.join(dataset_dir, split, 'images', final_img_name))
            else:
                continue

           
            txt_name = os.path.splitext(final_img_name)[0] + ".txt"
            txt_path = os.path.join(dataset_dir, split, 'labels', txt_name)
            
            with open(txt_path, 'w', encoding='utf-8') as txt_f:
                for shape in data.get('shapes', []):
                    label = shape['label']
                    if label not in class_map:
                        continue  
                    
                    class_id = class_map[label]
                    polygon_points = []
                    
                    # Normalize pixel coordinates into 0.0 - 1.0 range for YOLO
                    for pt in shape['points']:
                        x_norm = pt[0] / img_w
                        y_norm = pt[1] / img_h
                        polygon_points.append(f"{x_norm:.6f} {y_norm:.6f}")
                    
                    # Write YOLO Segmentation format: <class_id> <x1> <y1> <x2> <y2>
                    txt_f.write(f"{class_id} " + " ".join(polygon_points) + "\n")
        except Exception as e:
            print(f"Error processing file ({json_path}): {e}")
            continue

convert_and_distribute(train_files, 'train')
convert_and_distribute(val_files, 'val')

# Generate YAML configuration data
absolute_dataset_path = os.path.abspath(dataset_dir).replace("\\", "/")
yaml_content = f"""path: {absolute_dataset_path}
train: train/images
val: val/images

names:
  0: robot
  1: red_obj
  2: green_obj
  3: hand
"""

# Write dataset.yaml configuration file
yaml_path = os.path.join(dataset_dir, "dataset.yaml")
with open(yaml_path, 'w', encoding='utf-8') as yaml_f:
    yaml_f.write(yaml_content)

print("\n" + "="*60)
print("SUCCESS! Dataset conversion completed successfully.")
print(f"YOLO Configuration File Generated At: {yaml_path}")
print("="*60)