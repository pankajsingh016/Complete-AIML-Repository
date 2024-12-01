import streamlit as st
import pandas as pd
import numpy as np


#Title of the application 
st.title("Hello Streamlit")

# Display a Simple text
st.write("This is a simple text")

df = pd.DataFrame({
    "firs Column":[1,2,3,4,5],
    "second Column":[1,4,9,16,25]
})

#Display the dataframe
st.write("Here is the dataframe")
st.write(df)

## Create a line chart
chart_data = pd.DataFrame(np.random.randn(20,3),columns=['a','b','c'])
st.line_chart(df)

st.line_chart(chart_data)