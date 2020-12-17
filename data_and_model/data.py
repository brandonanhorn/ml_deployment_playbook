import pandas as pd
import numpy as np
import random

## creating a df with random numbers
anything = ["blue", "car", "house", "sport"]

df = pd.DataFrame(np.random.randint(0,51,size=(100, 4)), columns=list('ABCD'))

randomize = []
for i in range(100):
    random_choice = random.choice(anything)
    randomize.append(random_choice)

df["E"] = randomize

df.head(3)

## try to predict column d
target = "D"
X = df.drop(target, axis=1)
y = df[target]

## train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

## data prep
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn_pandas import DataFrameMapper

mapper = DataFrameMapper([
    (["A"], StandardScaler()),
    (["B"], StandardScaler()),
    (["C"], StandardScaler()),
    ("E", LabelBinarizer())], df_out=True)

## fit your data to new variables
Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)

# modelling with tensorflow
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential([
    Input(shape=(Z_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='elu')
])

model.compile(loss='mae', optimizer='adam')

history = model.fit(Z_train, y_train,
                    validation_data=(Z_test, y_test),
                    epochs=5, batch_size=1,
                    verbose=2)
