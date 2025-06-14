import streamlit as st # type: ignore
import numpy as np # type: ignore
import pickle
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore


#Load the LSTM Model
model = load_model('next_word_lstm.h5')


#Load the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer = pickle.load(handle)


## Function to predict
def predict_next_word(model,tokenizer, text, max_sequence_len):
    
    if not text.strip():
        return "Please Enter a Valid text"

    token_list = tokenizer.texts_to_sequences([text])[0]

    if not token_list:
        return "Unable to find any matching words in the tokenizer"

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]

    token_list = pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')

    predicted = model.predict(token_list,verbose=0)
    predicted_word_index = np.argmax(predicted,axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "Unable to predict the next word"


#streamlit app
st.title("NEXT WORD PREDICTION WITH LSTM AND EARLY STOPPING")
input_text = st.text_input("Enter the Sequence of Word")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f'Next Word:{next_word}')
    
 
