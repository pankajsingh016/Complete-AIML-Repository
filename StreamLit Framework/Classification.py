#importing the libraries

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species']=iris.target
    return df,iris.target_names 

#title
st.title("Iris Data-Set Classification")

#loading the dataset
df,target_name = load_data()
st.write(df)

#importing and fitting the model
model = RandomForestClassifier()
model.fit(df.iloc[:,:-1],df['species'])


#sidebar for slider
st.sidebar.title("Input Features")
sepal_length = st.sidebar.slider("Sepal Length",float(df['sepal length (cm)'].min()),float(df['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider("Sepal width",float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider("Petal length",float(df['petal length (cm)'].min()),float(df['petal length (cm)'].max()))
petal_width = st.sidebar.slider("Petal width",float(df['petal width (cm)'].min()), float(df['petal length (cm)'].max()))


input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
st.sidebar.write(input_data)

#Prediction
prediction = model.predict(input_data)

# st.write(prediction)
# st.write(target_name)
predicted_species = target_name[prediction[[0]]]

#Displaying the prediction
st.write("Prediction")
st.write(f"The predicted species is:{predicted_species}")

