# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 17:15:41 2022

@author: paart
"""

from flask import Flask,render_template,request
import dill
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)
symptoms = pd.read_csv("Symptom-severity.csv")['Symptom'].to_list()
new =pd.read_csv('new.csv')
y = new.iloc[:,0]
encoder =  LabelEncoder()
y1 = encoder.fit_transform(y)


model =joblib.load(open('model.joblib','rb'))
@app.route('/')
def helloworld():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    print(request.form)
    a = pd.DataFrame(columns=symptoms,index = [i for i in range(1) ])
    for i in request.form.values():
        a[0][i] = 1
        
    a.fillna(0,inplace = True)
    pred = model.predict(a)
    sorted_index_array = np.argsort(pred)
    sorted_array = a[sorted_index_array]
    rslt = sorted_array[-3 : ]
    rs = encoder.inverse_transform(rslt)
    return render_template('index.html',pred = rs[0] +rs[1] +rs[2] )

if __name__ == 'main' :
    app.run()














