# **Diffraction Optimization**

Project at the course **Modern Optics** (formerly Classical Optics) - TAU Spring 2023

## **Overview**

### **The Problem**
Design an optical system that performs classification. The input to the system is a field, which is composed of two elements:

1. An image of a digit, from the MNIST dataset
2. A *mask* function

The system needs to classify (optically) between images of the digits `0` and `1`.
The goal is to find a mask that **optimizes the accuracy of the classification**, using an any optimization method.

## **🛠 Development Setup**

Due to the `src/` layout of this project, in order to be able to write code, you must first install the project in it's virtual environment, using:
> `pip install -e .`

This wil resolve the issues with imports and allow development in _package_ mode.

## **📝 Dataset**

The dataset for this project is the MNIST dataset, specifically taken in `.csv` format from [Dariel Dato-On](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)'s submission to Kaggle.