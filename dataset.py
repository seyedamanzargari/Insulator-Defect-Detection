import shutil
import os
from sklearn.model_selection import train_test_split
import json
import cv2
from tqdm import tqdm
import requests
import json
import yaml
import xmltodict
import zipfile
import argparse
import random 


DIR = os.path.dirname(os.path.abspath(__file__))

# update this url form here https://ieee-dataport.org/competitions/insulator-defect-detection#files . copy Train_IDID_V1.2.zip url.
IDD_URL = 'https://ieee-dataport.s3.amazonaws.com/competition/78246/Train_IDID_V1.2.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20230814%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230814T122736Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=a258820c0e959482218c2fd5cd9feeb3ce829de686d275cbeb1ad86cdb432892'
CPLID_URL = 'https://github.com/InsulatorData/InsulatorDataSet/archive/refs/heads/master.zip'

def copy_files(source_dir, dest_dir):
    """Copy all files from source to destination directory."""
    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        
        if os.path.isfile(source_path):
            shutil.copy(source_path, dest_path)

def download_and_extract_idd_data():
    """Download and extract the IDD dataset."""
    
    print("Downloading IDD dataset...")
    response = requests.get(IDD_URL)
    zip_path = os.path.join(DIR, "IDD.zip")
    with open(zip_path, "wb") as f:
        f.write(response.content)
        
    print("Download complete.")
    
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(os.path.join(DIR, 'IDD'))
        
    print("Extraction complete.")

def convert_to_yolo(image_id, image_info, yolo_dir, img_width, img_height):
  """Convert bounding box to YOLO format and save to text file."""

  labels = []

  for obj in image_info['Labels']['objects']:
    if obj['string'] == 0:
      x, y, w, h = obj['bbox']

      # Normalize coordinates
      x_norm = x / img_width  
      y_norm = y / img_height
      w_norm = w / img_width
      h_norm = h / img_height

      # Calculate center x, y and width, height
      x_center = x_norm + w_norm / 2
      y_center = y_norm + h_norm / 2

      if 'No issues' in obj['conditions']:
        label = f"{0} {x_center} {y_center} {w_norm} {h_norm}"
        labels.append(label)

      if 'shell' in obj['conditions']:  
        label = f"{1} {x_center} {y_center} {w_norm} {h_norm}"
        labels.append(label)

  # Join labels and save to file
  label_text = "\n".join(labels)
  file_path = os.path.join(yolo_dir, "labels", f"{image_id}.txt")
  
  with open(file_path, 'w') as f:
    f.write(label_text)

def get_image_size(image_path):
  """Determine image size from file."""

  img = cv2.imread(image_path)
  height, width = img.shape[:2]

  return width, height

def prepare_idd_dataset():
  """Download, split and convert IDD dataset to YOLO format."""

  # Download and extract data
  if os.path.exists(os.path.join(DIR, 'IDD')) == False:
    download_and_extract_idd_data()

  # Load labels JSON
  with open(os.path.join(DIR, "IDD/Train/labels_v1.2.json"), "r") as f:
    data = json.load(f)
  
  dataset_folder = os.path.join(DIR, 'IDD/Train/Images')

  # Split data
  train_data, test_data = train_test_split(data, test_size=0.2, random_state=0)

  print(f"Number of training images: {len(train_data)}")
  print(f"Number of test images: {len(test_data)}")

  # Create output directories
  yolo_train_path = os.path.join(DIR, 'IDD_yolo_format/train')

  os.makedirs(f"{yolo_train_path}/images", exist_ok=True)
  os.makedirs(f"{yolo_train_path}/labels", exist_ok=True)

  yolo_test_path = os.path.join(DIR, 'IDD_yolo_format/test')

  os.makedirs(f"{yolo_test_path}/images", exist_ok=True)
  os.makedirs(f"{yolo_test_path}/labels", exist_ok=True)

  # Convert and save train data 
  # create train :
  for obj in tqdm(train_data):
    f_name = obj['filename']

    img_source = os.path.join(dataset_folder, f_name)
    img_destination = os.path.join(os.path.join(yolo_train_path, 'images'), f_name)

    if os.path.isfile(img_source):
        shutil.copy(img_source, img_destination)

        image_id = obj['filename'].split('.')[0]
        width, height = get_image_size(img_destination)
        convert_to_yolo(image_id, obj, yolo_train_path, width, height)

  # create test:
  for obj in tqdm(test_data):
    f_name = obj['filename']

    img_source = os.path.join(dataset_folder, f_name)
    img_destination = os.path.join(os.path.join(yolo_test_path, 'images'), f_name)

    if os.path.isfile(img_source):
        shutil.copy(img_source, img_destination)

        image_id = obj['filename'].split('.')[0]
        width, height = get_image_size(img_destination)
        convert_to_yolo(image_id, obj, yolo_test_path, width, height)

  # Create YAML config file
  dataset_config = {
    "train": yolo_train_path,
    "val": yolo_test_path,
    "nc": 2,
    "names": ["ok", "defect"],
    }

  with open(os.path.join(DIR, 'IDD_dataset.yaml'), 'w') as f:
    yaml.dump(dataset_config, f)

  print("IDD dataset preparation complete!")

def prepare_CPLID():
    response_site = requests.get(CPLID_URL)

    with open(os.path.join(DIR, 'CPLID.zip'), 'wb') as f:
        f.write(response_site.content)

    zip_ref = zipfile.ZipFile(os.path.join(DIR, "CPLID.zip"), 'r')
    zip_ref.extractall(os.path.join(DIR, "CPLID"))
    zip_ref.close()

    tmp_data_folder = os.path.join(DIR, 'CPLID_temp')
    tmp_folder_images = os.path.join(tmp_data_folder, 'images')
    tmp_folder_labels = os.path.join(tmp_data_folder, 'labels')
    os.makedirs(tmp_folder_images, exist_ok=True)
    os.makedirs(tmp_folder_labels, exist_ok=True)

    defective_insulator_path = os.path.join(DIR, 'CPLID/InsulatorDataSet-master/Defective_Insulators')
    normal_insulator_path = os.path.join(DIR, 'CPLID/InsulatorDataSet-master/Normal_Insulators')

    copy_files(os.path.join(normal_insulator_path, 'images'), tmp_folder_images)
    copy_files(os.path.join(defective_insulator_path, 'images'), tmp_folder_images)

    classes = ['insulator', 'defect']

    # normal insulator
    for img_name in tqdm(os.listdir(os.path.join(normal_insulator_path, 'images'))):
        label_path = os.path.join(os.path.join(normal_insulator_path, 'labels'), f'{os.path.basename(os.path.splitext(img_name)[0])}.xml')

        with open(label_path, "r") as f:
            xml_data = xmltodict.parse(f.read())

        img_width = int(xml_data['annotation']['size']['width'])
        img_height = int(xml_data['annotation']['size']['height'])

        yolo_anns = ''

        try:
            for obj in xml_data['annotation']['object']:
                clss = obj['name']
                box = obj['bndbox']
                
                x_min = int(box['xmin']) 
                y_min = int(box['ymin'])
                x_max = int(box['xmax'])
                y_max = int(box['ymax'])
                    
                x_center = (x_min + x_max) / 2 / img_width
                y_center = (y_min + y_max) / 2 / img_height
                w = (x_max - x_min) / img_width
                h = (y_max - y_min) / img_height
                    
                class_idx = classes.index(clss)
                    
                yolo_anns += f'{class_idx} {x_center} {y_center} {w} {h}\n'
        except:
            obj = xml_data['annotation']['object']
            clss = obj['name']
            box = obj['bndbox']
            
            x_min = int(box['xmin']) 
            y_min = int(box['ymin'])
            x_max = int(box['xmax'])
            y_max = int(box['ymax'])
                
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            w = (x_max - x_min) / img_width
            h = (y_max - y_min) / img_height
                
            class_idx = classes.index(clss)
                
            yolo_anns += f'{class_idx} {x_center} {y_center} {w} {h}\n'

        with open(os.path.join(tmp_folder_labels, os.path.splitext(img_name)[0] + '.txt'), 'w') as f:
            f.write(yolo_anns) 

    # defect insulator
    for img_name in tqdm(os.listdir(os.path.join(defective_insulator_path, 'images'))):
        insulator_label_path = os.path.join(os.path.join(defective_insulator_path, 'labels/insulator'), f'{os.path.basename(os.path.splitext(img_name)[0])}.xml')
        defect_label_path = os.path.join(os.path.join(defective_insulator_path, 'labels/defect'), f'{os.path.basename(os.path.splitext(img_name)[0])}.xml')
        
        with open(insulator_label_path, "r") as f:
            xml_data = xmltodict.parse(f.read())

        img_width = int(xml_data['annotation']['size']['width'])
        img_height = int(xml_data['annotation']['size']['height'])
        yolo_anns = ''
        try:
            for obj in xml_data['annotation']['object']:
                clss = obj['name']
                box = obj['bndbox']
                
                x_min = int(box['xmin']) 
                y_min = int(box['ymin'])
                x_max = int(box['xmax'])
                y_max = int(box['ymax'])
                    
                x_center = (x_min + x_max) / 2 / img_width
                y_center = (y_min + y_max) / 2 / img_height
                w = (x_max - x_min) / img_width
                h = (y_max - y_min) / img_height
                    
                class_idx = classes.index(clss)
                    
                yolo_anns += f'{class_idx} {x_center} {y_center} {w} {h}\n'
        except:
            obj = xml_data['annotation']['object']
            clss = obj['name']
            box = obj['bndbox']
            
            x_min = int(box['xmin']) 
            y_min = int(box['ymin'])
            x_max = int(box['xmax'])
            y_max = int(box['ymax'])
                
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            w = (x_max - x_min) / img_width
            h = (y_max - y_min) / img_height
                
            class_idx = classes.index(clss)
                
            yolo_anns += f'{class_idx} {x_center} {y_center} {w} {h}\n'

        with open(defect_label_path, "r") as f:
            xml_data = xmltodict.parse(f.read())

        img_width = int(xml_data['annotation']['size']['width'])
        img_height = int(xml_data['annotation']['size']['height'])

        try:
            for obj in xml_data['annotation']['object']:
                clss = obj['name']
                box = obj['bndbox']
                
                x_min = int(box['xmin']) 
                y_min = int(box['ymin'])
                x_max = int(box['xmax'])
                y_max = int(box['ymax'])
                    
                x_center = (x_min + x_max) / 2 / img_width
                y_center = (y_min + y_max) / 2 / img_height
                w = (x_max - x_min) / img_width
                h = (y_max - y_min) / img_height
                    
                class_idx = classes.index(clss)
                
                yolo_anns += f'{class_idx} {x_center} {y_center} {w} {h}\n'
        except:
            obj = xml_data['annotation']['object']
            clss = obj['name']
            box = obj['bndbox']
            
            
            x_min = int(box['xmin']) 
            y_min = int(box['ymin'])
            x_max = int(box['xmax'])
            y_max = int(box['ymax'])
                
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            w = (x_max - x_min) / img_width
            h = (y_max - y_min) / img_height
                
            class_idx = classes.index(clss)
                
            yolo_anns += f'{class_idx} {x_center} {y_center} {w} {h}\n'

        with open(os.path.join(tmp_folder_labels, os.path.splitext(img_name)[0] + '.txt'), 'w') as f:
            f.write(yolo_anns)

    dst_dataset_path = os.path.join(DIR, 'CPLID_yolo')

    train_yolo_path_labels = os.path.join(dst_dataset_path, 'train/labels')
    os.makedirs(train_yolo_path_labels, exist_ok=True)
    test_yolo_path_labels = os.path.join(dst_dataset_path, 'test/labels')
    os.makedirs(test_yolo_path_labels, exist_ok=True)
    train_yolo_path_images = os.path.join(dst_dataset_path, 'train/images')
    os.makedirs(train_yolo_path_images, exist_ok=True)
    test_yolo_path_images = os.path.join(dst_dataset_path, 'test/images')
    os.makedirs(test_yolo_path_images, exist_ok=True)

    image_files = os.listdir(tmp_folder_images)

    random.shuffle(image_files) 
    train_size = 0.8

    num_images = len(image_files)
    num_train = int(train_size*num_images)

    for i in tqdm(range(num_train)):
        train_image = image_files[i]
        shutil.copy(os.path.join(tmp_folder_images, train_image), train_yolo_path_images)
        shutil.copy(os.path.join(tmp_folder_labels, os.path.splitext(train_image)[0] + '.txt'), train_yolo_path_labels)

    for i in tqdm(range(num_train, num_images)):
        test_image = image_files[i]
        shutil.copy(os.path.join(tmp_folder_images, test_image), test_yolo_path_images)
        shutil.copy(os.path.join(tmp_folder_labels, os.path.splitext(test_image)[0] + '.txt'), test_yolo_path_labels)
    
    dataset_yaml = {
        'train': os.path.join(dst_dataset_path, 'train'), # folder with training images and labels
        'val': os.path.join(dst_dataset_path, 'test'), # folder with validation images and labels
        'nc': 2,
        'names': classes
        }

    with open('CPLID_dataset.yaml', 'w') as f:
        yaml.dump(dataset_yaml, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data", type=str, help="dataset name", default="IDD", choices=["IDD", "CPLID"])
    args = parser.parse_args()

    if args.data == "IDD":
        prepare_idd_dataset()
    elif args.data == "CPLID":
        prepare_CPLID()