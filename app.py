# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 13:04:31 2020

@author: Chinmay
"""

from flask import Flask,jsonify,render_template,request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
clf=pickle.load(open('Incident_management_model1.pkl','rb'))

@app.route('/')
def home():
    return render_temolate('index.html')

@app.route('/impact',methods=['POST'])
def impact():
    if request.method == 'POST':
        features=[val for val in request.form.values()]
        prediction= clf.predict(features)
        
        return render_template('index.html', prediction_text='Impact will be {}'.format(prediction))

if __name__== "__main__":
    app.run(debug=True)













