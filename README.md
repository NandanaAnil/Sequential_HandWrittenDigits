Handwritten Digit Recognition using ANN (Sequential Model)

This project demonstrates how to build and train a feedforward neural network (ANN/MLP) using TensorFlow/Keras to recognize handwritten digits from the MNIST dataset. The model is further tested on custom handwritten digit images after preprocessing.

📌 Project Overview

Loads the MNIST dataset (60,000 training and 10,000 testing images of digits 0–9).

Preprocesses images: flattens (28×28 → 784), normalizes pixel values, and one-hot encodes labels.

Builds a Sequential ANN model with multiple dense layers:

Hidden layers: 256 → 128 → 64 neurons (ReLU activation)

Output layer: 10 neurons (Softmax activation)

Trains the model with Adam optimizer and categorical crossentropy loss.

Evaluates performance using accuracy and a confusion matrix.

Predicts digits from external images (Zero.jpg, One.jpg, etc.) after preprocessing.

⚙️ Tech Stack

Python

TensorFlow / Keras

NumPy

PIL (Pillow)

Matplotlib

scikit-learn

🚀 How It Works

Training Phase

Train on MNIST for 10 epochs, batch size 128.

Validate with 10% of training data.

Achieve high accuracy on test data.

Evaluation

Test accuracy printed after evaluation.

Confusion matrix plotted to analyze misclassifications.

Custom Image Prediction

External digit images are preprocessed (grayscale, invert, resize to 28×28, normalize).

Trained model predicts digit.

Output image is displayed with predicted label.

📊 Results

High accuracy achieved on MNIST test data.

Confusion matrix highlights classification performance.

Model successfully predicts digits from external images.
