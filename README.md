# samsung-hand-gesture-recognition
Samsung Gesture Recognition Challenge using Tensorflow

1. gesture_recognition.py (ANN implementation in Tensorflow)  --> 50.26% (Test Accuracy)

2. gesture_recognition_CNN.py (CNN implementation in Tensorflow) --> 68.78% (Test Accuracy)

Converted all the images into binary images and then re-sized all images to a fixed dimension. Then used these processed images      for training my Convolutional Neural Network for gesture recognition(6 classes).
CNN Architecture used,
Input->Convolution->MaxPool->Relu->Conv->MaxPool->Relu->Flatten->FullConnected->Relu->FullConnected->Relu->SoftMax
I got a accuracy of 68.78% on the final test data.

