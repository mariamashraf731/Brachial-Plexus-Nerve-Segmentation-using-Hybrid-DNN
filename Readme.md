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


## Results and discussion

> we used accuracy as performance metric for both CNN and Unet and achieved 77% validation accuracy and 100% respectively.

##  Comparison with literature 

 Comparing to " J. Van Boxtel, V. Vousten, J. Pluim and N. M. Rad, "Hybrid Deep Neural Network for Brachial Plexus Nerve Segmentation in Ultrasound Images," 2021 29th European Signal Processing Conference (EUSIPCO), 2021, pp. 1246-1250, doi: 10.23919/EUSIPCO54536.2021.9616329. "  that used the hybrid model with different architecture and without transfer learning and achieved 72% (for U-net) we were able to achieve higher accuracy.




# Team
1. Ammar Al-Saeed Mohammed Section: 2, BN: 1.
2. Sohila Mohamed Maher    Section: 1, BN: 38.
3. Mariam Ashraf Mohamed   Section: 2, BN: 24