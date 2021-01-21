import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import tensorflow as tf
from tensorflow.keras.models import load_model

import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__,)
loaded_model = load_model("model.h5")

df = pd.read_csv("the_data.csv")

target = "D"
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y)

mapper = DataFrameMapper([
    (["A"], StandardScaler()),
    (["B"], StandardScaler()),
    (["C"], StandardScaler()),
    ("E", LabelBinarizer())], df_out=True)

pipe = make_pipeline(mapper, loaded_model)
pipe.fit(X_train, y_train)

new = pd.DataFrame({
    'A': [11],
    'B': [15],
    'C': [27],
    'E': "house"})

pipe.predict(new)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    args = request.form
    new = pd.DataFrame({
        'A': [args.get('A')],
        'B': [args.get('B')],
        'C': [args.get('C')],
        'E': [args.get('E')]})

    prediction = (pipe.predict(new)[0])
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(port=8080, debug=True)
