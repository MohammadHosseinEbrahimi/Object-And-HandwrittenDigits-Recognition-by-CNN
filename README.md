# Object and Handwritten Digits Recognition by Convolutional Neural Network
Deep learning convolutional neural network for recognition of CIFAR-10 image dataset and handwritten digits (UCI datasets)

**DATASET**

Handwritten digits: The dataset is comprised of 1,797 8×8 black&white photographs of handwritten digits from 0-9.
The data is from UCI ML hand-written digits datasets 
https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

Object detector: The dataset is comprised of 60,000 32×32 pixel color photographs of objects from 10 classes. 
Data is from  Canadian Institute For Advanced Research: 
https://www.cs.toronto.edu/~kriz/cifar.html

**DATASET PREPROCESSING**

Handwritten digits: The data is laready preprocessed. 

Object detector: We need to normalize the pixel values. 
Other preprocesses are already done.

**CNN model architecture** 

Object detector: model architecture includes 3 block of convolutional layers with small 3×3 filters followed by a max pooling layer with drop out layer. 
Drop out layer helps to avoid overfitting.
The number of filters are set arbitrarily to 32, 64 and 128. 
Padding 'SAME' ensures that filter is applied to all the elements of input similarly when filter does not perfectly fit the input image.
Each layer uses use the ReLU activation function.
For fully connected layers, first we flatten (2D to a vector) then we use kernel_regularizer to avoid overfitting.
For output layer, we have 10 output because we have 10 class to predict
To fit the model, we use 25 epoches to reach to the optimized values.
model architecture below:
![image](https://user-images.githubusercontent.com/109335350/186163100-7e0dc603-151d-446d-8545-5aeb402e1d02.png)
