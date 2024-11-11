import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## import ridge regressor and standard scaler pickle
linear_model = pickle.load(open('Linear-Regression-Ecommerce/linear_model.pkl', 'rb'))
standard_scaler = pickle.load(open('Linear-Regression-Ecommerce/scaler.pkl', 'rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict_price',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        avg_session_length = float(request.form.get('avg_session_length', 0))
        time_on_app = float(request.form.get('time_on_app', 0))
        time_on_website = float(request.form.get('time_on_website', 0))
        length_of_membership = float(request.form.get('length_of_membership', 0))

        new_data_scaled = standard_scaler.transform([[avg_session_length, time_on_app, time_on_website, length_of_membership]])
        result = linear_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0")