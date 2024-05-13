import numpy as np
import streamlit as st
from PIL import Imagecd MR
from utils import process_image, get_footer, load_model
import keras.preprocessing.image as image
from copy import deepcopy
import os
import cv2

colors = ["red", "orange", "green", "brown"]
options = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
def highlight_prediction(options, idx):
    options = deepcopy(options)
    highlight = f'''<span style="color:{colors[idx]}" >**{options[idx]}** </span> '''
    options[idx] = highlight
    return '<br>'.join(options)


if __name__ == '__main__':

    st.markdown("<h1 style='text-align: center; color: black;'>"
                "<center>&emsp;&emsp;Alzheimer Stage Detection  from Medical Images</center></h1>",
                unsafe_allow_html=True)
    st.markdown("<br>Developed by Nithin ", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a  medical image...", type=["jpg"])

    st.markdown(get_footer(), unsafe_allow_html=True)

    if uploaded_file is not None:

        img_in = Image.open(uploaded_file)

        # Resize the image to the expected model input shape (128, 128)
        expected_shape = (128, 128)
        img_array = np.array(img_in)
        img_in_processed = cv2.resize(img_array, expected_shape)

        col1, col2 = st.columns(2)
        col1.image(img_in_processed)
        st.write("Brain MRI Image")

        # Load the model with the same version of Keras
        model = load_model('MRI.h5')
        prediction = model.predict(img_in_processed[np.newaxis, ...])  # Add a new axis for batch dimension
        idx = np.argmax(prediction)
        col2.markdown("### Predicted  Type")

        col2.markdown(highlight_prediction(options, idx), unsafe_allow_html=True)

        st.image("assets/symptoms.png")
    else:

        st.image("assets/symptoms.png")
