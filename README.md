# Image-Captions---Transfer-Learning-for-Resnet

# COCO dataset

## The Common Objects in Context (COCO) dataset is a large-scale dataset designed for object detection, segmentation, and captioning. Created by Microsoft in 2014.
### Key Statistics

330K images
1.5M object instances
80 object categories
5 captions per image
250,000+ people with keypoints

### Tasks

Object Detection
Instance Segmentation
Keypoint Detection
Panoptic Segmentation
Image Captioning

### Dataset Splits

Train: 118K images
Validation: 5K images
Test: 41K images

### Data Format

Annotations in JSON format
Contains bounding boxes, segmentation masks, keypoints
Multiple categories per image
Detailed captions describing scene content

### Common Use Cases

Computer vision research
Model benchmarking
Deep learning applications
Transfer learning

### Example imaghe + caption:

![image](https://github.com/user-attachments/assets/1500cfe6-29d0-403a-a899-3b5d8a09259d)


## This project

A deep learning model that generates natural language descriptions for images using PyTorch. The model combines a pre-trained ResNet-50 encoder with an LSTM decoder for caption generation.
Architecture

Encoder: Modified ResNet-50 pre-trained on ImageNet

Final fully connected layer replaced to output desired embedding size
Feature extraction layers frozen for transfer learning
Includes batch normalization, dropout, and ReLU activation


Decoder: LSTM-based sequence generator

Embedding layer for word representation
LSTM with configurable hidden size and layers
Fully connected output layer for vocabulary prediction
Support for teacher forcing during training

### Implementation Details
**Data Preprocessing**

```ruby
transform = Compose([
    Lambda(lambda img: img.convert("RGB")),
    Resize((224, 224)),
    ToTensor(),  
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**Training**

Loss: Cross-Entropy with padding token ignored
Optimizer: Adam with learning rate 0.001
Gradient clipping to prevent exploding gradients
Best model checkpoint saving
Teacher forcing for sequence generation

**Inference**

Greedy decoding strategy
Support for batch processing
Special token handling and caption cleaning
Image preprocessing pipeline
