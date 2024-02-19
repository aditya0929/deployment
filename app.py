
import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np

# Load the pre-trained model
model_path = "/content/drive/MyDrive/Pneumonia.h5"
model = tf.keras.models.load_model(model_path)

# Function to preprocess the image for prediction
def preprocess_image(image):
    # Convert BytesIO object to Image
    image = Image.open(image)
    
    # Resize and normalize the image
    img_array = cv2.resize(np.array(image), (224, 224)) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Function to make predictions
def predict_pneumonia(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit app
def main():
    st.title("Pneumonia Detection App")
    st.sidebar.title("Upload Image")

    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.sidebar.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Make prediction
        prediction = predict_pneumonia(uploaded_file)

        # Display the prediction
        if prediction[0][0] > 0.5:
            st.error("Pneumonia Detected!")
        else:
            st.success("No Pneumonia Detected!")

if __name__ == '__main__':
    main()

