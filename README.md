# Generative AI & Probabilistic Models - B.Tech Assignments

This repository contains practical laboratory experiments and assignments focusing on Generative Artificial Intelligence and Probabilistic Modeling. It includes complete, executable Jupyter Notebooks utilizing PyTorch, TensorFlow, and Hugging Face Transformers.

## 📁 Repository Structure

### 1. Assignment 1: Generative AI & GMMs (`Assignment_1_Generative_AI_and_GMM.ipynb`)
- **Creative Content Generation:** Utilizes the GPT-2 language model from Hugging Face for automated text generation.
- **Probabilistic Modeling:** Implements Gaussian Mixture Models (GMM) using synthetic datasets to demonstrate probabilistic clustering, distribution fitting, and density estimation.

### 2. Assignment 2: Autoencoders & Variational Autoencoders (`Assignment_2_Autoencoders_and_VAE.ipynb`)
- **Part A - Convolutional Autoencoder (AE):** Implements a Convolutional AE from scratch using PyTorch. Trained on the MNIST dataset to perform robust image denoising and dimensionality reduction (compressing 784-dimensional images into a 2D latent space manifold).
- **Part B - Variational Autoencoder (VAE):** Implements a VAE using the reparameterization trick and KL Divergence. Trained on the Labeled Faces in the Wild (LFW) dataset to map human facial features into a continuous normal distribution, allowing for the generation of novel faces and smooth latent-space interpolations between distinct facial structures.

### 3. Assignment 3: Generative Adversarial Networks (`Assignment_3_GANs.ipynb`)
- **Deep Convolutional GAN (DCGAN):** Implements a fully adversarial network from scratch using PyTorch. Trained on the CIFAR-10 dataset to implicitly learn the distribution of real-world objects and generate novel, synthetic images from pure random noise. Includes an analysis of the Generator and Discriminator loss equilibrium, visualizations of mode evolution across epochs, and a concluding discussion on the ethics of deepfakes and real-world generative applications.

## 🚀 How to Run Locally

We highly recommend using a virtual environment (via `venv` or `conda`) to run these notebooks locally. Alternatively, these notebooks are fully optimized to be uploaded and run seamlessly in **Google Colab** (GPU runtime recommended for the VAE).

### 1. Setup Virtual Environment
```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate
# Activate it (macOS/Linux)
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Dependencies include `torch`, `torchvision`, `transformers`, `scikit-learn`, `matplotlib`, `numpy`, and `jupyter`)*

### 3. Launch Jupyter Notebook
```bash
jupyter notebook
```
From the browser interface, open `Assignment_1_Generative_AI_and_GMM.ipynb`, `Assignment_2_Autoencoders_and_VAE.ipynb`, or `Assignment_3_GANs.ipynb` and run the cells interactively!
