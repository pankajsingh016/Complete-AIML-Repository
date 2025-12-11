import streamlit as st
import pandas as pd
import numpy as np

## Title of application
st.title("Hello Streamlit")


# display a simple text
st.text("demo text")
st.write('This is a simple text')

# create and print the dataframe
df = pd.DataFrame({
    'first column':[1,2,3,4],
    'second column':[10,20,30,40]
})

# Display the dataframe
st.write("here is the dataframe")
st.write(df)


# create a line chart
st.line_chart(df)

chart_data = pd.DataFrame(np.random.randn(40,3),columns=['a','b','c'])
st.line_chart(chart_data)


## Widgets in streamlit

st.title('Learning Widgets in Streamlit')
name = st.text_input("Enter your name:", placeholder='Pankaj')
if name:
    st.write(f'Hello {name}')
age = st.slider('Select your age',0,100,25)
st.text('Select Box:')
options = ['Python','Java','C++','JavaScript']
choice = st.selectbox('Choose Your Favourrite Language:',options)
st.write(f'You Selected {choice}')

if 'data_student' not in st.session_state:
    st.session_state.data_student = pd.DataFrame(columns=['Name','Age','Subject'])

def funkonclick(name,age,choice):
    new_row = {'Name':name,'Age':age,'Subject':choice}
    st.session_state.data_student.loc[len(st.session_state.data_student)] = new_row 

st.button('Add',on_click=funkonclick,args=(name,age,choice))

st.write(st.session_state.data_student)