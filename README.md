# Face-recognition
This application implements a facial recognition system of 10 well known people: Adele, Angelina Jolie, Arnold Schwarzenegger, Bon Jovi, Brad Pitt, Chuck Norris, Conor Mcgregor, Cristiano Ronaldo, Ed Sheeran and Eddie Murphy. The users are allow to create their own convolutional neural network by choosing parameters for each layer in the model from the GUI interfance or to load a pre-trained model. 

## Pre-trained model
The pre-trained was trained on on a dataset containing 200 pictures of each person. The model accuracy was 88.75% on a training set.

## Face detection
For the purpose of the face detection was used the Haar Cascade Clasifier. The file of this clasifier is included in this repository. 

## Requirements
- Python 3.7
- Cv2
- Numpy
- Pickle
- TensorFlow
- PyQt5

The easiest way to run the project is to create an anaconda environment and install the required packages. You also need to download and extract files from the Important part of README into the repository directory.


## Important
This repository has not the dataset and the pre-trained model files. These files do not meet the maximum file standards on github. You can get the model from: https://1drv.ms/u/s!Au8HZ076-lBxgi6f7ViFkoy1sZzK?e=6TcPcc
