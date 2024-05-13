from tensorflow.keras import models
from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf

from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.optimizers import SGD
import cv2


def load_model(model_path):
    model = models.load_model(model_path )
    return model

def process_image(im, desired_size=254):
    
    
    im = im.resize((desired_size,) * 2, resample=Image.LANCZOS)
    #im=im.resize(128,128)
    im = np.expand_dims(np.array(im),0)
    return im

def get_footer():
    
    footer = """
     <style>
    footer {
	visibility: hidden;
	}
    footer:after {
       
	content:' '; 
	visibility: visible;
	display: block;
	position: relative;
	padding: 5px;
	top: 2px;
	
    }
    </style>
    """
    return footer
