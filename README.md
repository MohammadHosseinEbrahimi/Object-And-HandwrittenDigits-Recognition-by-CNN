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

Handwritten digits: The data is already preprocessed. 

Object detector: We need to normalize the pixel values. 
Other preprocesses are already done.

**CNN model architecture** 

Handwritten digits: Due to simplicity of this task, we use a model architecture including a convolutional layer with small 3×3 filters, default padding='valid', (no padding so the input image gets fully covered), default strides=(1, 1), i.e., filter step shifts over the input matrix, followed by a max pooling layer (to reduce the computational cost) without drop out layer (no regularization). 
The number of filters can be set arbitrarily to 32.
Each layer uses use the ReLU activation function.


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

**evaluting the model** 



Object detector: The graph below shows the model has been trained well enough. The accuracy on train set reached a plateau.
The accuracy on train set was 0.94, and on test set was 0.81.

![image](https://user-images.githubusercontent.com/109335350/186378308-6eb43a5d-04b7-4542-b624-a8a5c7729b17.png)



**A random sample to check the model performance**


Object detector: 

![image](https://user-images.githubusercontent.com/109335350/186379645-cb0cf362-348b-49d5-bd8c-0c617dfb36ca.png)

