import tensorflow as tf
from keras import Model
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.utils.image_utils import load_img, img_to_array
import numpy as np

# Model
from tensorflow.keras.models import load_model
model = load_model("braintumor.h5")
classes = ["glioma","meningioma", "No Tumor", "pituitary"]

# Creating Api
from fastapi import FastAPI
import streamlit as st

App = FastAPI()

def Brain_UI():
  st.title("Brain Tumor Detection ")
  file = st.file_uploader("Input Patient MRI Image ")
  ok = st.button("Predict")
  try:
    if ok == True: # if user pressed ok button then True passed
      img = load_img(file, target_size=(224, 224))
      x = img_to_array(img)
      x = x * (1. / 255)
      x = np.expand_dims(x, axis=0)
      result = classes[np.argmax(model.predict(x))]
      print("The patient have",result)
      # classindx = predict_image(im1)
      if result == "glioma":
        st.error("Tumor type is glioma")
      elif result == "meningioma":
        st.error("Tumor type is meningioma")
      elif result == "No Tumor":
        st.success("Patient has No Tumor")
      elif result == "pituitary":
        st.error("Tumor type is pituitary")
  except Exception as e:  # all error
    st.info(e)

if __name__ == "__main__":
  import uvicorn

  uvicorn.run(App, host="0.0.0.0", port=8000, log_level='info')
