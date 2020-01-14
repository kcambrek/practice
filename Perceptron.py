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
seed = 42
data_form = st.selectbox("Choose data form:", ["Vanilla", "Blobs", "Circles"])
n_samples = st.slider("Amount of data points", 2, 200)
distortion = st.slider("Distortion", 0.0, 1.0, step = 0.01, format = "%.2f")

#Generating data
if data_form == "Vanilla":
    x, y = datasets.make_classification(n_samples = n_samples, n_features = 2, n_redundant = 0, flip_y = distortion, random_state = seed)

if data_form == "Blobs":
    x, y = datasets.make_blobs(n_samples = n_samples, n_features = 2, centers = 2, cluster_std = distortion * 10, random_state = seed)
    
if data_form == "Circles":
    x, y = datasets.make_circles(n_samples = n_samples, noise = distortion, random_state = seed)

x = (x - x.min()) / (x.max() - x.min())*10

epochs = st.slider("Epochs", 1, 100)
lr = st.slider("Learning rate", 0.001 , 1.0, step = 0.001, format = "%.3f")


class Perceptron:
    '''Class for a 2_D perceptron.'''
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self.b = np.array([0, 0])
        self.c = 0
        
    def classify(self, x):
        activation = (self.b*x).sum(axis = 1)+self.c
        return [1 if i >=0 else 0 for i in activation]
    
    def create_boundary(self):
        b1 = self.b.reshape(-1)[0]
        b2 = self.b.reshape(-1)[1]
        if self.c == 0:
            self.c = 0.00001
        y = (-(self.c / b2) / (self.c / b1))*x[:,0] + (-self.c / b2)
        return(y)
    
    def train(self, x, y):
        for c, i in enumerate(x):
            xi = i.reshape(1, -1)
            predicted_i = self.classify(xi)
            real_i = y[c]
            self.b = self.b + self.lr * (real_i - predicted_i) * xi
            self.c = self.c + self.lr * (real_i - predicted_i)

def accuracy_check(y, y_hat):
   return sum([i == j for i,j in zip(y, y_hat)])/len(y)
    
perceptron = Perceptron(lr, epochs)        


y_hat = perceptron.classify(x)
df = pd.DataFrame({"Predicted" : y_hat, "Y" : y, "X1" : x[: , 0], "X2" : x[:, 1]})


epoch = st.header("Epoch: "+ str(0))


#build classification plot
base = alt.Chart(data = df).encode(alt.X("X1:Q"))
z1 = base.mark_circle().encode(y = alt.Y("X2:Q"), color = "Y:N")
plot = z1
progress_plot = st.altair_chart(plot)

accuracy = accuracy_check(perceptron.classify(x), y)
score = st.header("Accuracy: "+ str(accuracy))

#build error plot
accuracy_df = pd.DataFrame({"Epoch" : [0], "Accuracy" : [accuracy]})
plot = alt.Chart(accuracy_df).mark_line(color = "blue").encode(alt.X("Epoch:Q"), alt.Y("Accuracy:Q"))
accuracy_plot = st.altair_chart(plot)

#every epoch, the prediction- and accuracy plot, and the weights are updated.
for i in range(epochs):
    epoch.header("Epoch: "+ str(i))
    score.header("Accuracy: "+ str(accuracy_check(perceptron.classify(x), y)))
    boundary = perceptron.create_boundary()
    df["Boundary"] = boundary
    base = alt.Chart(data = df).encode(x = alt.X("X1:Q"))
    z1 = base.mark_circle().encode(y = alt.Y("X2:Q"), color = "Y:N")
    z2 = base.mark_line(color = "red").encode(y = alt.Y("Boundary:Q"))
    plot = z1 + z2
    progress_plot.altair_chart(plot.interactive())
    perceptron.train(df[["X1", "X2"]].values, y)
    
    accuracy = accuracy_check(perceptron.classify(x), y)
    accuracy_df = pd.DataFrame({"Epoch" : [i], "Accuracy" : [accuracy]})
    accuracy_plot.add_rows(accuracy_df)
    
    time.sleep(0.1)
    


    

