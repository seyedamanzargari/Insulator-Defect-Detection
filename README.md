# Insulator Defect Detection Dataset Preparation

First we have to prepares the [IDD](https://ieee-dataport.org/competitions/insulator-defect-detection) and [CPLID](https://github.com/InsulatorData/InsulatorDataSet) insulator image datasets for object detection by converting the annotations to YOLO format.

#### What the Code Does

- Downloads the IDD and CPLID insulator image datasets
- Splits the datasets into train and test sets
- Converts the XML annotations to YOLO format by calculating normalized bounding box coordinates
- Copies the images and YOLO annotations to train and test folders
- Creates a YAML configuration file for each dataset for use with YOLO models

#### How the Code Works
- Uses the requests and zipfile libraries to download and extract the datasets
- Loads the JSON or XML annotations
- Calculates the YOLO bounding box coordinates from the annotation data
- Writes the YOLO annotations to text files named for each image
- Copies the images to train and test folders, splitting randomly
- Creates a YAML file with paths, class names and number of classes

### How to Use

The code dependencies are listed in requirements.txt. To install:<br>
```pip install -r requirements.txt```

The code can then be run with:<br>
```python dataset.py -d [dataset]```

Where dataset is either `IDD` or `CPLID`.

This will download, prepare and split the specified dataset.<br>
The output will be train and test folders for each dataset in YOLO format, along with a YAML configuration file.<br>
The prepared datasets can then be used to train YOLO models for insulator defect detection.

# Train YOLO Model for Insulator Defect Detection

After data prepration we have to trains a YOLOv8 object detection model on the IDD or CPLID datasets.

#### What the Code Does
- Loads a YOLOv8 model
- Trains the model on either the IDD or CPLID dataset
- Outputs the trained model weights and metrics

### How to Use
Then to train:<br>
```python train.py -d [dataset]```

Where `[dataset]` is either `IDD` or `CPLID`.<br>
This will load the respective dataset YAML file and begin training.

The major training parameters:

    -m: YOLOv8 model size - n, s, m, l, x
    -b: Batch size
    -e: Number of epochs
    -s: Input image size
    -pretrained: Fine-tuning
    --device: CPU, GPU or MPU
    --name: Name of project

Default is YOLOv8n model trained for 30 epochs at batch size 32 on 640x640 images.<br>
The trained model weights will be saved to `runs/train/exp/weights/best.pt`.<br>
Metrics and training logs will be in Tensorboard and logged to the `runs/train/exp` folder.

# YOLO Model Training Results

### Results on CPLID
The nano model achieves higher precision, recall and mAP than the small model throughout training.

Nano model:
- Reaches 98% mAP@50 after 10 epochs
- 75% mAP@50-95 after 10 epochs

Small model:
- Reaches 98% mAP@50 after 10 epochs
- 78% mAP@50-95 after 10 epochs

The nano model converges faster and achieves better COCO metrics compared to the small model.

### Results on IDD
Similarly for IDD, the nano model outperforms the small model.

Nano model:
- 85% mAP@50 after 10 epochs
- 65% mAP@50-95 after 10 epochs

Small model:
- 90% mAP@50 after 10 epochs
- 69% mAP@50-95 after 10 epochs

The nano model has higher recall throughout training, indicating it is better at finding objects. The small model has slightly higher precision at epochs 9 and 10.
Overall the nano model achieves better performance on IDD as well.

In summary, for both datasets, the nano model converges faster and achieves better object detection performance compared to the small model. 

Maybe the higher resolution provides more discriminative features to the model.

# Make Predictions with Trained YOLO Insulator Defect Model
#### What the Code Does
- Loads a trained YOLO model
- Runs the model on an input image to detect defects
- Displays the image with bounding boxes around detected defects

### How to Use
To make predictions after training a model:<br>
`python predict.py -m [model] -s [image]`

Where:

    [model]: Path to trained model weights file, e.g. best.pt
    [image]: Path to input image to run inference on

This will run the model on the image and display the results with OpenCV.<br>
Detected defects will have bounding boxes drawn around them along with class label.

# Make Predictions with Trained YOLO Model through Gradio App
This code provides a web interface for inferencing using a trained YOLO insulator defect detection model with [Gradio](https://www.gradio.app/).

#### What the Code Does
- Creates a Gradio interface that takes in a model and image
- Runs inferencing on the image using the YOLO model
- Returns the image with bounding boxes drawn around detected defects

#### Gradio Installation
Gradio can be installed with pip:<br>
```pip install gradio```

### How to Use
To launch the web interface:

```python predict_panel.py```

This will start a local web server. Navigate to the printed URL.<br>
Upload an image and enter the path to a trained YOLO model weights file.<br>
Hit submit to run inferencing and see the results!


