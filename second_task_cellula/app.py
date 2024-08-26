import os
import numpy as np
import streamlit as st 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = load_model(r'C:\Users\DELL\Downloads\IBM\first_task_cellula\cellula_first_task_model.h5')

# Define the folder-to-label mapping
folder_to_label = {'CaS': 0, 'CoS': 1, 'Gum': 2, 'MC': 3, 'OC': 4, 'OLP': 5, 'OT': 6}
label_to_folder = {v: k for k, v in folder_to_label.items()}  # Reverse mapping for display

def classify_image(img):
    """Classify the uploaded image using the pre-trained ResNet50 model."""
    img = img.resize((256, 256))  # Resize the image to match the input size of the model
    img_array = image.img_to_array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the model's input shape
    img_array /= 255.0  # Normalize the image

    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction, axis=1)[0]
    return label_to_folder[predicted_label]

def main():
    st.title("Teeth Disease Classifier")
    st.write("Upload an image of a tooth to classify it.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        
        if st.button("Classify"):
            result = classify_image(img)
            st.success(f'The image is classified as: {result}')
    
    if st.button("About"):
        st.text("Teeth Disease Classification")
        st.text("Built with Streamlit and ResNet50")

if __name__ == '__main__':
    main()
