# Brain Tumor Segmentation with UNETR on the BraTS2021 Dataset

Welcome to my project repository! I developed a brain tumor segmentation pipeline using the UNETR model on the BraTS2021 dataset. I created this project as part of an assignment, leveraging MONAI, and various preprocessing utilities to train and evaluate a deep learning model for medical image segmentation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Key Components](#key-components)
  - [Assignment Details](#assignment-details)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training and Execution](#model-training-and-execution)
- [Files Overview](#files-overview)
- [Installation and Requirements](#installation-and-requirements)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

I focused on segmenting brain tumors from multi-modal MRI scans using the UNETR (U-Net Transformer) model. By leveraging transfer learning and advanced data augmentation techniques, I aimed to accurately delineate tumor regions in 3D MRI volumes. This project allowed me to experiment with medical image segmentation methods in a research setting.

---

## Dataset

For this project, I used the **BraTS2021** dataset, which includes:
- **MRI Modalities**: T1, T1Gd, T2, and FLAIR images.
- **Labels**: Segmentation masks that delineate tumor regions.

I organized the dataset using a JSON metadata file (`dataset_metadata.json`) that splits the data into training, validation, and (optionally) test sets.

---

## Key Components

### Assignment Details

The assignment required me to:
- Analyze the shape, voxel range, and labels of each MRI modality.
- Visualize 3D MRI slices and their corresponding segmentation masks.
- Implement the UNETR model for brain tumor segmentation.
- Modify the dataloader and transformation pipeline to suit the BraTS dataset.
- Train the model (even for a few epochs) to demonstrate a complete segmentation workflow.

For more details, you can refer to the assignment document: [assessment_assignment.pdf](./assessment_assignment.pdf).

### Data Preprocessing

In the **data_utils.py** script, I handled:
- Loading images and labels using MONAI transforms.
- Applying a series of preprocessing steps such as intensity scaling, orientation adjustment, spatial resampling, cropping, and data augmentation.
- Creating data loaders that support distributed training and caching to improve efficiency.

### Model Training and Execution

The **main.py** script is my entry point for training the UNETR model. It:
- Parses command-line arguments for configuration (e.g., learning rate, batch size, number of epochs).
- Loads the dataset using the preprocessing pipeline.
- Instantiates the UNETR model (imported from `networks.unetr`) with appropriate parameters.
- Initiates training using a custom training loop, along with loss functions (e.g., DiceCELoss), metrics (e.g., DiceMetric), and learning rate scheduling.

I also provided a Colab notebook (`BraTS2021_UNETR.ipynb`) that demonstrates the interactive development and experimentation process.

---

## Files Overview

- **assessment_assignment.pdf**: The assignment document that outlines the project requirements and objectives.
- **data_utils.py**: Contains utilities for data loading, preprocessing, and transformation using MONAI.
- **dataset_metadata.json**: Organizes the dataset file paths and splits for training, validation, and testing.
- **main.py**: The main script for executing the training pipeline and evaluating the model.
- **BraTS2021_UNETR.ipynb**: A Colab notebook that illustrates the workflow from data exploration to model training and visualization.

---

## Installation and Requirements

### Prerequisites

- Python 3.7 or higher
- PyTorch
- MONAI and its related dependencies (e.g., NumPy, Torchvision)
- Additional libraries: argparse, numpy, matplotlib

