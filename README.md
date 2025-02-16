# Sea Trash Detection

## Overview
Sea Trash Detection is a deep learning project designed to identify and classify marine debris in images. The model is trained using a labeled dataset to enhance the accuracy of detection and segmentation.

## Features
- Deep learning-based image classification for marine debris
- Comprehensive dataset for training and validation
- Automated data preprocessing and augmentation
- Model evaluation and prediction functionalities

## Installation
### Prerequisites
Ensure you have Python installed. You can create a virtual environment for dependency management:
```bash
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```

### Clone the Repository
```bash
git clone https://github.com/your-repo/sea-trash-detection.git
cd sea-trash-detection
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Dataset
The dataset used for training contains images of marine debris and other underwater objects. It is automatically downloaded in the code. Ensure the dataset follows this structure:
```
/trash_inst_material
  /train/images
  /val/images
```

### Dataset Download
The dataset is directly included in the code and will be downloaded automatically when the script runs.

### Class Labels
The dataset consists of the following categories:
- **rov**
- **plant**
- **animal_fish**
- **animal_starfish**
- **animal_shells**
- **animal_crab**
- **animal_eel**
- **animal_etc**
- **trash_etc**
- **trash_fabric**
- **trash_fishing_gear**
- **trash_metal**
- **trash_paper**
- **trash_plastic**
- **trash_rubber**
- **trash_wood**

## Labeling using Roboflow
The dataset was labeled using [Roboflow](https://roboflow.com/). Follow these steps if you want to label additional data:
1. **Create a Roboflow Account**: Sign up at [Roboflow](https://roboflow.com/).
2. **Upload Images**: Create a new project and upload your dataset images.
3. **Annotate Images**: Use Roboflow's annotation tool to label objects manually or use its auto-labeling features.
4. **Generate Dataset**: Once labeling is complete, export the dataset in your preferred format (e.g., YOLO, COCO, Pascal VOC).
5. **Download and Use**: Download the labeled dataset and integrate it into your project for training.

## Usage
Run the Jupyter Notebook to train and evaluate the model:
```bash
jupyter notebook Sea_trash_detection.ipynb
```

## Model
The model is based on a Convolutional Neural Network (CNN) designed for image classification. The pipeline includes:
- Data preprocessing and augmentation
- Model training and optimization
- Evaluation using test data

## Results
The trained model outputs predictions that highlight detected trash objects in images, aiding in marine debris identification and environmental cleanup efforts.

## License
This project is open-source and licensed under the [MIT License](LICENSE).


