from flask import Flask, render_template
from flask import request
from sklearn.externals import joblib
import pickle

import numpy as np
app=Flask(__name__)
#app=Flask(__name__)
#@app.route('/test')
#def test():
    #return "Flask is been used for development"

#Load model_prediction

@app.route('/')
def home():
    return render_template('home.html')
@app.route("/predict",methods=['GET','POST'])
def predict():
    if request.method=='POST':
        try:
            temperature=float(request.form['temperature'])
            humidity=float(request.form['humidity'])
            city=float(request.form['city'])
            month=float(request.form['month'])
            pred_args=[temperature,humidity,city,month]
            pred_args_arr=np.array(pred_args)
            pred_args_arr=pred_args_arr.reshape(1,-1)
            mul_model=open("multiple_model.pkl","rb")
            ml_model=joblib.load(mul_model)
            model_prediction=ml_model.predict(pred_args_arr)

        except ValueError:
            return "please check if value are entered correctly"
    return render_template('predict.html',prediction=model_prediction)
if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0')
