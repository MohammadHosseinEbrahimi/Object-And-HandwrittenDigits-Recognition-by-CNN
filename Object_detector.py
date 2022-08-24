"""
This code uses convolutional neural network for predicting a set of objects

The dataset is comprised of 60,000 32×32 pixel color photographs of objects 
from 10 classes. Dataset was intended for computer vision research.
Data is from  Canadian Institute For Advanced Research: 
https://www.cs.toronto.edu/~kriz/cifar.html

Labels are as below:
0: airplane
1: automobile
2: bird
3: cat
4: deer
5: dog
6: frog
7: horse
8: ship
9: truck

Written by Mohammadhossein Ebrahimi
22.08.2022
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from tensorflow.keras.datasets import cifar10 #Downloading the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#Normalizing the pixel values
X_train = X_train.astype('float32')/ 255.0
X_test = X_test.astype('float32')/ 255.0

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from keras.regularizers import l2
#defining model
model=Sequential()
# model architecture includes 3 block of convolutional layers with small 3×3 
# filters followed by a max pooling layer with drop out layer. 
# Drop out layer helps to avoid overfitting
# The number of filters are set arbitrarily to 32, 64 and 128. 
# Padding 'SAME' ensures that filter is applied to all the elements of input 
# similarly when filter does not perfectly fit the input image (corner pixels
# cotribute same as centeral ones).
# Each layer uses the ReLU activation function.
model.add(Conv2D(32,(3,3),activation='relu', padding='same', input_shape=(32,32,3)))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.2)) # dropout layer for regularization
model.add(Conv2D(64, (3, 3), padding='same', activation='relu' ))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.2))# dropout layer for regularization
model.add(Conv2D(128, (3, 3), padding='same', activation='relu' ))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.2))# dropout layer for regularization

# adding fully connected layers, first we flatten (2D to a vector)
# initial tests showed signs of overfitting, so we use kernel_regularizer.
model.add(Flatten())
model.add(Dense(128,activation='relu',kernel_regularizer=l2(0.001)))
# adding output layer, 10 because we have 10 class to predict
model.add(Dense(10,activation='softmax'))
# compiling the model
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',
              metrics=['accuracy'])

# fitting the model, number of epoches must be large enough to reach to
# the optimized values
Model_fit= model.fit(X_train,y_train,epochs=100)

# evaluting the model
Accuracy_train= model.evaluate(X_train,y_train)
Accuracy_test= model.evaluate(X_test,y_test)

# summarize history for accuracy
import matplotlib.pyplot as plt
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(Model_fit.history['accuracy'], 'g-')
ax2.plot( Model_fit.history['loss'], 'b-')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy', color='g')
ax2.set_ylabel('loss', color='b')
plt.show()

# A few random samples to check the model
import numpy as np
for sample in np.random.randint(1,len(X_test), size=5):
  # Generate a plot
  fig = plt.figure()
  plt.imshow(X_test[sample],cmap = plt.cm.gray_r)
  txt = "True class is %d"%y_test[sample]
  txt2 = "Predicted class is %d"%np.argmax(model.predict(X_test),axis=1)[sample]
  fig.text(-0.1,0.1,txt)
  fig.text(-0.1,0.15,txt2)
  plt.show()
 