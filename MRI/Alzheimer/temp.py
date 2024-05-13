import numpy as np
import streamlit as st
from PIL import Image
from utils import process_image, get_footer, load_model
import keras.preprocessing.image as image
from copy import deepcopy
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

colors = ["red", "orange", "green", "brown"]
options = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']


def highlight_prediction(options, idx):
    options = deepcopy(options)
    highlight = f'''<span style="color:{colors[idx]}" >**{options[idx]}** </span> '''
    options[idx] = highlight
    return '<br>'.join(options)


if __name__ == '__main__':
    st.image("D:\\Nithin\\LB\\FRSH\\Project\\Dementia\\CODE\\MRI\\Alzheimer\\Assets\\brain.jpeg")
    st.write(" ")
    st.markdown("<h1 style='text-align: center; color: black;'>"
                "<center>&emsp;&emsp;Alzheimer Stage Detection  from Medical Images</center></h1>",
                unsafe_allow_html=True)
    st.markdown("<br>Developed", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a  medical image...", type=["jpg"])

    st.markdown(get_footer(), unsafe_allow_html=True)

    if uploaded_file is not None:

        img_in = Image.open(uploaded_file)

        img_in_processed = process_image(img_in)

        col1, col2 = st.columns(2)
        col1.image(img_in_processed)
        st.write("Brain MRI Image")

        model = load_model('Trained_Models/LastModel2.h5')
        prediction = model.predict(img_in_processed)
        idx = np.argmax(prediction)
        col2.markdown("### Predicted  Type")

        col2.markdown(highlight_prediction(options, idx), unsafe_allow_html=True)

        st.image("assets/symptoms.png")
    else:

        st.image("assets/symptoms.png")

