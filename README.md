# Face-recognition
This application implements a facial recognition system of 10 well known people: Adele, Angelina Jolie, Arnold Schwarzenegger, Bon Jovi, Brad Pitt, Chuck Norris, Conor Mcgregor, Cristiano Ronaldo, Ed Sheeran and Eddie Murphy. The users are allow to create their own convolutional neural network by choosing parameters for each layer in the model from the GUI interfance or to load a pre-trained model. 

## Application

![alt text](https://github.com/pytocabra/Face-recognition/blob/main/app.png)

## Requirements
- Python 3.7
- Cv2
- Numpy
- Pickle
- TensorFlow 2.x
- PyQt5

## Pre-trained model
The pre-trained was trained on on a dataset containing 200 pictures of each person. The model accuracy was 88.75% on a training set.
The model is based on 4 Conv2D layers with 64, 64, 128 and 256 filters respectively. 

## Face detection
For the purpose of the face detection was used the Haar Cascade Clasifier. The file of this clasifier is included in this repository. 



