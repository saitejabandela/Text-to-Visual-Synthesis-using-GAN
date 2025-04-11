# Text-to-Image Synthesis using GAN

# Introduction
This project explores the intersection of Natural Language Processing (NLP) and Computer Vision by converting detailed text descriptions into high-resolution images using 
Stacked Generative Adversarial Networks. The challenge lies in the accurate and detailed visualization of text descriptions, which is addressed through a two-stage StackGAN model. The project utilizes the Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset, known for its detailed classification of bird species.

# Project Highlights
Utilization of StackGAN for detailed and high-resolution image generation from text descriptions.

Implementation of a two-stage process for image synthesis, enhancing detail and accuracy.

Comprehensive data preprocessing and model training to improve image quality.

Exploration of Generative Adversarial Networks (GANs) in the context of text-to-image synthesis.

# Dataset
Caltech-UCSD Birds-200-2011 (CUB-200-2011) contains 200 categories, 11788 images

Annotations per image include 15 Part Locations, 312 Binary Attributes, and 1 Bounding Box

# Dependencies
Python 3.x

TensorFlow or PyTorch

NumPy

Matplotlib (for visualization)

Pre-trained text embeddings (available from the reedscot repository)

# Setup & Installation
Clone the repository: git clone <repo-url>

Install the required dependencies: pip install -r requirements.txt

Download the CUB-200-2011 dataset and place it in the data/ directory.

# Usage
Preprocess the dataset: python preprocess.py

Train the StackGAN model: python train.py

Generate images from text descriptions: python generate.py --text "Your text description here"

# Results

Inception Score: 1.0742203, indicating room for improvement in image diversity and realism.

Generated images showcase the potential of StackGAN in text-to-image synthesis.

# Future Work

Enhance the model to generate more realistic images.

Incorporate advanced text encoders for more accurate text-to-image synthesis.

Adapt the model for specific domains, such as fashion or healthcare.

# Contact
For any queries or further discussion, please reach out to: rithwik.maramraju@sjsu.edu
