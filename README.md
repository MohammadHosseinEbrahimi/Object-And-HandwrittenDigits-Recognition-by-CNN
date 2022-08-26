# Object and Handwritten Digits Recognition by Convolutional Neural Network
Deep learning convolutional neural network for recognition of CIFAR-10 image dataset and handwritten digits (UCI datasets)

**Dataset**

Handwritten digits: The dataset is comprised of 1,797 8×8 black&white photographs of handwritten digits from 0-9.
The data is from UCI ML hand-written digits datasets 
https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

Object detector: The dataset is comprised of 60,000 32×32 pixel color photographs of objects from 10 classes. 0: airplane
1: automobile
2: bird
3: cat
4: deer
5: dog
6: frog
7: horse
8: ship
9: truck.
Data is from  Canadian Institute For Advanced Research: 
https://www.cs.toronto.edu/~kriz/cifar.html

**Dataset Preprocessing**

Handwritten digits: The data is already preprocessed. 

Object detector: We need to normalize the pixel values. 
Other preprocesses are already done.

**CNN model architecture** 

Handwritten digits: Due to simplicity of this task, we use a model architecture including a convolutional layer with small 3×3 filters, default padding='valid', (no padding so the input image gets fully covered), default strides=(1, 1), i.e., filter step shifts over the input matrix, followed by a max pooling layer (to reduce the computational cost) without drop out layer (no regularization). 
The number of filters can be set arbitrarily to 32.
Each layer uses the ReLU activation function.

![image](https://user-images.githubusercontent.com/109335350/186381368-f4e9951a-303a-4ad4-9bb0-f90869b2cea7.png)


Object detector: model architecture includes 3 block of convolutional layers with small 3×3 filters followed by a max pooling layer with drop out layer. 
Drop out layer helps to avoid overfitting.
The number of filters are set arbitrarily to 32, 64 and 128. 
Padding 'SAME' ensures that filter is applied to all the elements of input similarly when filter does not perfectly fit the input image.
Each layer uses the ReLU activation function.
For fully connected layers, first we flatten (2D to a vector) then we use kernel_regularizer to avoid overfitting.
For output layer, we have 10 output because we have 10 class to predict.
To fit the model, we use 100 epoches to reach to the optimized values.
model architecture below:

![image](https://user-images.githubusercontent.com/109335350/186163100-7e0dc603-151d-446d-8545-5aeb402e1d02.png)

**Evaluating the model** 

Handwritten digits: The graph below shows the model has been trained well enough. The accuracy on train set reached a plateau.
The accuracy on train set was 0.99, and on test set was 0.98.

![image](https://user-images.githubusercontent.com/109335350/186380469-f0053d6b-9054-47ae-81dd-3f8d4da1effd.png)


Object detector: The graph below shows the model has been trained well enough. The accuracy on train set reached a plateau.
The accuracy on train set was 0.92, and on test set was 0.83.

![image](https://user-images.githubusercontent.com/109335350/186420258-7f461d64-dc5c-4956-9a20-c2521ad339ad.png)


**A random sample to check the model performance**

Handwritten digits:

![image](https://user-images.githubusercontent.com/109335350/186380811-68d525c4-bc0d-47a7-8293-993b02928ef7.png)


Object detector: 

![image](https://user-images.githubusercontent.com/109335350/186379645-cb0cf362-348b-49d5-bd8c-0c617dfb36ca.png)

