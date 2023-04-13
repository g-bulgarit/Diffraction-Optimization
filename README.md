# **Diffraction Optimization**

Project at the course **Modern Optics** (formerly Classical Optics) - TAU Spring 2023

## **🔍Overview**

### **The Problem**

Design an optical system that performs classification. The input to the system is a field, which is composed of two elements:

1. An image of a digit, from the MNIST dataset
2. A *mask* function

The system needs to classify (optically) between images of the digits `0` and `1`.
The goal is to find a mask that **optimizes the accuracy of the classification**, using an any optimization method.

### **Method**

We had to pick a way to go from an initial phase mask to the one that maximizes our success in classification of MNIST digits.
Some possible approaches:
- ML methods, specifically Deep Learning
- Classical methods, like Genetic Algorithms

We decided to try and utilize **Simulated Annealing** as our optimization algorithm. 

In essence it's a search algorithm over the search space, but it utilizes a `temperature` parameter to reduce the delta between soltions over time, similar to the relative freedom of an atom in a metal grid to move during the initial (hot) stage of the annealing process.

Just like in nature, as the system cools the atoms can move less and they settle into a position of minimal potential energy. The algorithm mimcs this behavior by initially allowing big jumps over the search space (temperature is hot -> energy is high). This way, it's more likely that the solution will avoid local minima. Over time it the system temperature is lowered, and with it the probability to perform big jumps is lessened, allowing system to settle onto a good solution which is less likely to be a *local* minima.

### **Results**
The system is capable of distinguishing between `0` and `1` images with an accuracy of over 90%.


## **🛠 Development Setup**

Due to the `src/` layout of this project, in order to be able to write code, you must first install the project in it's virtual environment, using:
> `pip install -e .`

This wil resolve the issues with imports and allow development in _package_ mode.

## **📝 Dataset**

The dataset for this project is the MNIST dataset, specifically taken in `.csv` format from [Dariel Dato-On](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)'s submission to Kaggle.