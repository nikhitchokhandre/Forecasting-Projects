# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 12:36:27 2023

@author: Mr.Nikhit
"""


from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    # Load the data and preprocess it
    data = pd.read_csv('oil.csv', parse_dates=['Date'], index_col='Date')
    data = data.resample('D').mean().interpolate()

    # Fit the ARIMA model to the data
    model = ARIMA(data, order=(3,1,2))
    results = model.fit()

    # Generate predictions for the input dates
    predictions = results.predict(start=start_date, end=end_date, dynamic=True)

    # Format the predictions as an HTML table
    table = pd.DataFrame({'Date': predictions.index, 'Price': predictions}).to_html(index=False)

    # Generate a line chart of the predictions
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Price'], label='Actual')
    ax.plot(predictions.index, predictions, label='Predicted')
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Oil Price Prediction')
    
    chart_path = 'static/chart.png'
    fig.savefig(chart_path)
    plt.close(fig)
    
    return render_template('index.html', table=table,chart_path = chart_path)

if __name__ == '__main__':
    app.run(debug=False)