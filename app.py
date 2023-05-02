import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array


st.set_page_config(page_title="Cancer Prediction", layout="centered", initial_sidebar_state="auto", menu_items=None)

loaded_model = tf.keras.models.load_model('lungcane.h5')


def prediction(image):
    img=load_img(image,target_size=(64,64))
    test_image=img_to_array(img)
    test_image=np.expand_dims(test_image,axis=0)
    result=loaded_model.predict(test_image)
    
    if result[0][0]==0:
        return ("Predicted to be Cancerous",img)
    else: 
        return ("Predicted to be Non-Cancerous",img)

    
col1, col2, col3 = st.columns([2, 8, 2])
col2.image("cancer.jpeg", use_column_width=True)
    
upload_img = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if upload_img != None:
        col1, col2, col3 = st.columns([3, 13, 2])
        predict = col1.button("Predict")
        clr_btn= col3.button("Clear")
        
        if clr_btn:
            st.cache_resource()
            
        if predict:
            cancer, show_img = prediction(upload_img)
            
            html_str = f"""
            <style>
            p.a {{
            font: bold 25px Source Sans Pro;
            }}
            </style>
            <br>
            <center>
            <p class="a">{cancer}</p>
            </center>
            """
            st.markdown(html_str, unsafe_allow_html=True)
            col1, col2, col3 = st.columns([3, 5, 3])
            col2.image(show_img, use_column_width=True)