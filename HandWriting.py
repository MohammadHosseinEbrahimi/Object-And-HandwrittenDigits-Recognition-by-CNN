"""
This code uses convolutional neural network for handwriting digits recognition

The dataset is comprised of 1,797 8×8 black&white photographs of 
handwritten digits from 0-9.
The data is from UCI ML hand-written digits datasets
https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits


Written by Mohammadhossein Ebrahimi
19.08.2022
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random

digits = datasets.load_digits()

# To randomly check and visualize an example
Random_number= random.randint(1,50)
fig = plt.figure()
plt.imshow(digits.images[Random_number],cmap = plt.cm.gray_r)
txt = "This is %d"%digits.target[Random_number]
fig.text(0.1,0.1,txt)
plt.show()

X_train, X_test, y_train, y_test = train_test_split (
    digits.images, digits.target, random_state=0)

#reshaping data
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]
                           , 1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1)) 


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
#defining model
model=Sequential()
# Due to simplicity of this task, we use a model architecture including  
# a convolutional layer with small 3×3 filters, default padding='valid', 
# (no padding so the input image gets fully covered) 
# defult strides=(1, 1), i.e., filter step shifts over the input matrix
# followed by a max pooling layer (to reduce the computational cost).
# without drop out layer (no regularization). 
# The number of filters can be set arbitrarily to 32. 
# Each layer uses use the ReLU activation function.
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(8,8,1)))
#adding pooling layer, it will cut the image dimension by half
model.add(MaxPool2D(2,2))
#adding fully connected layer (feature detector part of the model), 
# first we flatten it to 1D vector
model.add(Flatten())
model.add(Dense(100,activation='relu'))
#adding output layer
model.add(Dense(10,activation='softmax'))
#compiling the model
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',
              metrics=['accuracy'])
#fitting the model
Model_fit= model.fit(X_train,y_train,epochs=10) # batch_size is defult = 32 (mini-batch), 
# epochs=10 seems enough because it is a rather simple model, default learning 
# rate is 0.01 with no momentum (variant of the stochastic gradient descent)

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


# A few random samples to check it
for sample in np.random.randint(1,len(X_test), size=5):
  # Generate a plot
  fig = plt.figure()
  plt.imshow(X_test[sample],cmap = plt.cm.gray_r)
  txt = "This is %d"%y_test[sample]
  txt2 = "predicted %d"%np.argmax(model.predict(X_test),axis=1)[sample]
  fig.text(-0.1,0.1,txt)
  fig.text(-0.1,0.15,txt2)
  plt.show()
 
#acuracy of a dummy classifier (uniform class)
#This is to ensure that our actual model makes sense
from sklearn.dummy import DummyClassifier
Dumm= DummyClassifier(strategy="uniform")
Dumm.fit(X_train, y_train)


print('Accuracy of dummy on test set: {:.2f}'
     .format(Dumm.score(X_test, y_test)))
print('Accuracy of CNN on test set: {:.2f}'
     .format(model.evaluate(X_test, y_test)[1]))


