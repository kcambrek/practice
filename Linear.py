# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 16:35:50 2020

@author: kees.brekelmans
"""
from sklearn import datasets
import numpy as np
import random
import altair as alt
import pandas as pd
import streamlit as st
import time

#(hyper)parameters set before learning
epochs = st.slider("Epochs", 1, 100)
lr = st.slider("Learning rate", 0.001 , 0.04, step = 0.001, format = "%.3f")
noise = st.slider("Noise", 0, 100)
seed = 42


#Generating data
x, y = datasets.make_regression(n_samples = 50, n_features = 1, noise = noise, random_state = seed)
x = (x - x.min()) / (x.max() - x.min())*10
y = (y - y.min()) / (y.max() - y.min())*100
x = x.reshape(-1)


def cost_function(predicted, true):
    '''Returns the MSE'''
    return np.sum(np.square(predicted-true))/predicted.shape[0]
    


class Linear_regression:
    '''Class for the linear regression model. Currently works only with one dependent variable.'''
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self.b = 0
        self.c = 0       
        
    def find_derivatives(self, x, predicted, true):
        b_d = (-2/predicted.shape[0] * sum((true - predicted) * x))
        c_d = (-2/predicted.shape[0] * sum(true - predicted))
        return b_d, c_d

    def train(self, x, y):
        predicted = self.predict(x)
        b_d, c_d = self.find_derivatives(x, predicted, y)
        self.b = self.b - b_d * self.lr
        self.c = self.c - c_d * self.lr

    def predict(self, x):
        return self.b*x+self.c

model = Linear_regression(lr, epochs)

epoch = st.header("Epoch: "+ str(0))


#build prediction plot
y_hat = model.predict(x)
df = pd.DataFrame({"Predicted" : y_hat, "Y" : y, "X" : x})
base = alt.Chart(data = df).encode(alt.X("X:Q"))
z1 = base.mark_line(color = "blue").encode(y = 'Predicted:Q')
z2 = base.mark_circle(color = "red").encode(y = "Y:Q")
plot = z1 + z2
progress_plot = st.altair_chart(plot)

model_parameters = st.info("y = {:.2f}x + {:.2f}".format(model.b, model.c))

#build error plot
error = cost_function(model.predict(x), y)
error_df = pd.DataFrame({"Epoch" : [0], "Loss" : [error]})
plot = alt.Chart(error_df).mark_line(color = "red").encode(alt.X("Epoch:Q"), alt.Y("Loss:Q"))
cost = st.header("Error: "+ str(error))
error_plot = st.altair_chart(plot)

#every epoch, the prediction- and error plot, and the weights are updated.
for i in range(epochs):
    epoch.header("Epoch: "+ str(i))
    error = cost_function(model.predict(x), y)
    cost.header("Error: "+ str(error))
    y_hat = model.predict(x)
    model.train(x, y)
    model_parameters.info("y = {:.2f}x + {:.2f}".format(model.b, model.c))
    
    df = pd.DataFrame({"Predicted" : y_hat, "Y" : y, "X" : x})
    error_df = pd.DataFrame({"Epoch" : [i], "Loss" : [error]})
    
    base = alt.Chart(data = df).encode(alt.X("X:Q"))
    z1 = base.mark_line(color = "blue").encode(y = 'Predicted:Q')
    z2 = base.mark_circle(color = "red").encode(y = "Y:Q")            
    plot = z1 + z2
    
    progress_plot.altair_chart(plot.interactive())
    error_plot.add_rows(error_df)
    time.sleep(0.1)
    