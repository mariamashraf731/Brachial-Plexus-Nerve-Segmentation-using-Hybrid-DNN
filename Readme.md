# 🩺 Brachial Plexus Nerve Segmentation (Hybrid DNN)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange)
![Technique](https://img.shields.io/badge/Technique-Transfer%20Learning-green)
![Model](https://img.shields.io/badge/Model-ResNet50%20%2B%20U--Net%20(VGG16)-purple)
![Domain](https://img.shields.io/badge/Domain-Medical%20Imaging-red)

## 👥 Team
1.  **Ammar Al-Saeed Mohammed** (Section: 2, BN: 1)
2.  **Sohila Mohamed Maher** (Section: 1, BN: 38)
3.  **Mariam Ashraf Mohamed** (Section: 2, BN: 24)

## 📌 Project Overview
This project implements a **Deep Learning based Semantic Segmentation** solution to locate the **Brachial Plexus Nerve** in ultrasound images. Locating this nerve accurately is crucial for doctors to perform nerve block anesthesia procedures effectively, reducing reliance on narcotics.

### 🧠 The Hybrid Approach
Instead of running segmentation on every image, we propose a **Two-Stage Pipeline**:
1.  **Classification Stage:** Filter images to check if the nerve is present or not.
2.  **Segmentation Stage:** Apply segmentation only on images where the nerve is detected.

## 📂 Dataset
* **Source:** Ultrasound Nerve Segmentation competition on Kaggle.
* **Content:** 120 ultrasound images of the neck from 47 subjects.
* **Preprocessing:**
    * **Channel Broadcasting:** Converting grayscale ultrasound images to 3-channel RGB.
    * **Resizing:** Standardizing input dimensions for the pre-trained models.

## 🛠️ Network Architecture & Tech Stack

### 1️⃣ Stage 1: Classification (Nerve Presence Detection)
* **Model:** **ResNet50** (Pre-trained on ImageNet).
* **Configuration:**
    * **Input:** 224x224 RGB Images.
    * **Head:** Fully Connected Layer (1024 units, ReLU) $\rightarrow$ Output Layer (2 units, Sigmoid).
    * **Optimization:** Adam Optimizer ($lr=0.001$), Batch size = 64.
* **Purpose:** To classify images into `Nerve Present` vs `Nerve Absent`.

### 2️⃣ Stage 2: Semantic Segmentation
* **Model:** **U-Net** with **VGG16 Encoder**.
* **Configuration:**
    * **Encoder (Downsampling):** VGG16 (Pre-trained on ImageNet, weights frozen).
    * **Decoder (Upsampling):** Transpose Convolution layers + Batch Normalization + Concatenation (Skip Connections).
    * **Optimization:** Adam Optimizer ($lr=0.001$), Batch size = 10.
* **Purpose:** To generate a precise mask locating the nerve in positive images.

## 📊 Results & Discussion
We utilized **Transfer Learning** to boost performance and reduce training time given the limited dataset size.

| Model Stage | Architecture | Metric | Result on Validation |
|:-----------:|:------------:|:------:|:--------------------:|
| **Classifier** | ResNet50 | Accuracy | **77%** |
| **Segmentor** | VGG16 U-Net | Pixel Accuracy | **~99%** |

### 🏆 Comparison with Literature
We compared our results with:
> *J. Van Boxtel, et al., "Hybrid Deep Neural Network for Brachial Plexus Nerve Segmentation in Ultrasound Images," EUSIPCO 2021.*

The referenced paper achieved **72% accuracy** for their U-Net implementation without transfer learning. By employing **Transfer Learning** with VGG16 as a backbone, our model demonstrated superior performance and faster convergence.

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/mariamashraf731/Hybrid-Nerve-Segmentor.git](https://github.com/mariamashraf731/Hybrid-Nerve-Segmentor.git)
    ```
2.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Notebook:**
    Open `notebooks/Hybrid_Approach_Pipeline.ipynb` to train and evaluate the pipeline.

---

# Brachial Plexus nerve segmentation from Ultrasound images
>in this project we try to build a DL model that can do semantic segmentation to locate the Brachial Plexus nerve using US images. the main application is helping doctors locate this nerve in a fast and efficient manner using US, to be able to use the nerve in a nerve block anasthesia procedure rather than relying on narcotics .
---
## Dataset
> The dataset is downloaded from ultrasound nerve segmentation competion on kaggle.  
it contains 120 ultrasound images of neck for 47 subjects.
---
## Preprocessing 
for preprocessing, the following steps are implemented:
1. Broadcasting image channels to convert from grey scale to RGB.
2. resize
---
## Network architecture
> our model consists of 2 networks:
1.  CNN to classify images according to brachial plexus nerve presence as present or unpresent. 
    * We selected only the test images that were classified as containing nerves to test the U-net.
2. U-Net for semantic segmentation for images where the nerve is present.
> the layers in each network are as follows:
1. CNN: Resnet50 + Fully connected layer + output layer
2. U-net:  Encoder->VGG16 , Decoder->Transpose convultion layers + batchnorm

## Hyperparameters
1) CNN -> Batchsize=64 , epochs=25 ,lr=0.001, optimizer:Adam
2) Unet -> Batchsize=10 , epochs=10 ,lr=0.001, optimizer:Adam

## Transfer learning
> we employed transfer learning as an attempt to boost performance and reduce number of trainable parameters.
1. CNN: we imported the weights of resnet50 network.
2. U-Net: we used VGG16 pretrained network as the encoder and added  new trainable layers for the decoder . 
