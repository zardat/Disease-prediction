# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
# from tensorflow.keras.optimizer import SGD
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.utils import np_utils
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
# from sklearn.pipeline import Pipeline
# import pickle
# import dill
# import weakref
# import joblib


new =pd.read_csv('new.csv')
x = new.iloc[:,1:134]
y = new.iloc[:,0]
# dataset = pd.read_csv("dataset.csv")
# lis = dataset.columns.to_list()
# for i in lis:
#     dataset[i] = dataset[i].str.strip()
# lis.remove('Disease')
# symptoms = pd.read_csv("Symptom-severity.csv")['Symptom'].to_list()
# disease = pd.read_csv("symptom_precaution.csv")["Disease"]
# empty = pd.DataFrame(columns=symptoms,index = [i for i in range(4920) ])
# data = pd.concat([dataset,empty],axis = 1)


# for ind in dataset.index:
#     for i in range(17):
#         if data.loc[ind][i] != np.nan :
#             data.loc[ind][data.loc[ind][i]] = 1

# data.fillna(0,inplace =True)
# data = data.drop(lis,axis = 1)
# data.to_csv("new.csv",index = False) 

encoder =  LabelEncoder()
y1 = encoder.fit_transform(y)
Y = pd.get_dummies(y1).values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.2, random_state=0)


model = Sequential()
model.add(Dense(128, input_dim=133, activation='relu'))
model.add(Dense(41, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

md = model.fit(X_train, y_train, batch_size=50, epochs=50)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

y_pred = model.predict(X_test)


# actual = np.argmax(y_test,axis=1)
# predicted = np.argmax(y_pred,axis=1)
# print(f"Actual: {actual}")
# print(f"Predicted: {predicted}")

# print(f'Training Set Accuracy: {(predicted == actual).mean() * 100:f}')

# pickle.dump(md,open('important', 'wb'),)
# hack36 = pickle.load(open('model.pkl','rb'))
# # mode = pickle.dumps(md,open('important', 'wb'))
# dill.dumps(model, open('file.pkl', 'wb'))




