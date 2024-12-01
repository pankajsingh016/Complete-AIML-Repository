import streamlit as st
import numpy as np
import pandas as pd

st.title("Streamlit Text Input")

#input box in streamlit
name = st.text_input("Enter Your Name:")

#slider in streamlit
age = st.slider("Select Your age:",0,100,25)


#select box in streamlit
options = ['Python','Java','C','C++','Rust','Go']
choice = st.selectbox("Choose your favorie language:",options)

if name:
    st.write(f"Hello, {name}")

if choice:
    st.write(f"Opting for {choice} language.")