# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 06:01:53 2020

@author: HOME
"""

import pandas as pd
import numpy as np
data=pd.read_csv('iris.csv').values
x=data[:,:-1]
y=data[:,len(data[0])-1]
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(64,input_dim=4,activation='relu'))
model.add(Dense(120,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(3,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
from keras.utils import np_utils
new_y=np_utils.to_categorical(y)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,new_y,test_size=0.1)
model.fit(x_train,y_train,epochs=100,validation_split=0.1)
from matplotlib import pyplot as plt
plt.plot(model.history.history['loss'],label='loss')

plt.xlabel('#epochs')
plt.ylabel('loss')
#we should have test data to do this prediction
predicted_target=model.predict(test_data)
print("predicted target")

