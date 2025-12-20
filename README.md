# Handwritten Word Recognition - Project for Artificial Intelligence
## Handwritten word recognition using two datasets and visualizing with PyGame
This project implements a handwritten character and word recognition system using Convolutional Neural Networks (CNNs). The system recognizes uppercase letters (A–Z) and digits (0-9), using two separate datasets and models.

The trained models are used in a PyGame application, where users can draw a character, digit or word and get a prediction for what is written on the canvas.

### Datasets
- [A–Z dataset](letters/A_Z%20Handwritten%20Data.csv) is a dataset that contains around 372 000 rows of uppercase letters only. It is in a CSV format, where the first column is the label (0-25 for A-Z), and the rest of the columns (1-784) are flattened 28×28 pixels (0-255).
- [0-9 dataset](numbers) is a MNIST dataset that contains 70 000 grayscale 28×28 images of handwritten digits 0-9 from around 250 writers. It is split as 60 000 train / 10 000 test, with around 7 000 samples per class. The original binary format uses .idx files loaded via idx2numpy. The pixels are normalized from 0-255 to 0-1.

### Training of two separate models
In this [notebook](Separate Models.ipynb) there is training for two models, one for the uppercase letters, and another one for the digits using CNN.<br>
After the training, the models are saved in separate files:
- [Letter Model](letter_cnn_26cls.pth)
- [Digit Model](digit_cnn_10cls.pth)

### Visualization in PyGame
1. [Single Character Drawing](letter_draw_predict.py)
- Draw one letter or one digit on a canvas
- The drawing is converted into a pixel grid
- The grid is flattened and passed to the CNN
- Displays the predicted character on screen

2. [Word Drawing](word_draw_predict.py)
- Draw multiple characters sequentially to form a word
- Each character is isolated and preprocessed
- Each character is classified independently
- Predictions are combined into a final word


## Technologies Used
- Python
- PyTorch
- Pandas (CSV handling)
- NumPy
- Pygame
- Jupyter Notebook

## How to Run
1. Train the models using the notebooks (or download the saved models).
2. Place the .pth model files in the models/ directory.
3. Run one of the Pygame scripts:
```
python letter_draw_predict.py
```
or
```
python word_draw_predict.py
```
