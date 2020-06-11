from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np

import pickle

# load the model from the disk
loaded_model=pickle.load(open('Stock Price.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
     return render_template('home.html')
 
@app.route('/predict',methods=['POST'])
def predict():
    df=pd.read_csv('C:/Users/Lenovo/Desktop/Project/Stock-Price-Prediction-using-Keras-and-Recurrent-Neural-Networ-master/Google_Stock_Price_Test.csv')
    my_prediction=loaded_model.predict(df.iloc[:,:-1].values)
    my_prediction=my_prediction.tolist()
    return render_template('result.html',prediction=my_prediction)



if __name__ == '__main__':
     app.run(debug=True)


