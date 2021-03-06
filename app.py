# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1w_Zjcj7jZ7eVgE7EShE9ejOQ_lakK1Ib
"""

#loading libraries
import joblib
from flask import Flask, request, jsonify, render_template
from get_prediction import get_recommendations
import pickle


import flask
app = Flask(__name__)
model= pickle.load(open('catboost_v3.pkl','rb'))
# render_template

@app.route('/')
def home():

    """Serve homepage template."""
    return render_template('index.html')
    

@app.route('/predict', methods=['POST'])
def predict():

    to_predict_list = request.form.to_dict()
    #print(to_predict_list)
    predictions, time = get_recommendations(to_predict_list)
    print(predictions, time)
    if 'recommend' not in predictions.keys():
        
        return render_template("new_user_recommendation.html",predictions = predictions)

    return render_template("predict.html",predictions = predictions)
    #return jsonify({'products': recommended_products, 'Time': difference, 'predict_list':to_predict_list, 'top5':top5_products})
    
    
if __name__ == "__main__":
    app.run(debug=True)
