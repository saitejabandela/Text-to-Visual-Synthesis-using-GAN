# Text-to-Visual-Synthesis-using-GAN

 **Overview**
This project bridges the fields of Natural Language Processing (NLP) and Computer Vision by transforming rich text descriptions into realistic, high-resolution images. Leveraging a two-stage Stacked Generative Adversarial Network (StackGAN), the system synthesizes bird images based on textual input, showcasing the powerful potential of GANs in understanding and visualizing language.

We used the CUB-200-2011 dataset, which contains fine-grained annotations for bird species, to train and evaluate our model. This project demonstrates how deep generative models can be used to visualize textual semantics with increasing detail and accuracy.

**Key Features**
- StackGAN Implementation: Employs a two-phase GAN architecture for refining image detail and resolution progressively.

- End-to-End Pipeline: Includes data preprocessing, text embedding, training, and image generation modules.

- High-Quality Synthesis: Generates 256x256 resolution images from descriptive text inputs.

- Exploration of GANs: Applies generative modeling in the NLP-to-vision domain using an interpretable architecture.

- Dataset: CUB-200-2011 (Caltech-UCSD Birds)
200 bird categories with a total of 11,788 images.

Each image includes:

15 annotated part locations

312 binary attributes

1 bounding box

Ideal for fine-grained visual tasks like this one.

**Dependencies**
Ensure the following are installed:

Python 3.x

TensorFlow or PyTorch

NumPy

Matplotlib

Pre-trained text embeddings (from reedscot repository)
