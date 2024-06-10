import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load features and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Modify model architecture
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Nut And Blot Cassification  System')

# Rust detection function
def detect_rust(img_path):
    try:
        # Load the image
        img = Image.open(img_path)
        # Convert image to RGB mode (if it's in RGBA mode)
        img = img.convert("RGB")
        # Resize image to a smaller size for faster processing (if needed)
        img = img.resize((224, 224))
        # Convert image to numpy array
        img_array = np.array(img)
        # Check if any pixels fall within the rust color range
        rust_pixels = np.sum((img_array >= [100, 0, 0]) & (img_array <= [255, 100, 100]), axis=-1)
        rust_percentage = np.count_nonzero(rust_pixels) / rust_pixels.size * 100
        if rust_percentage > 1:  # You can adjust this threshold as needed
            rust_detected = True
        else:
            rust_detected = False
        return rust_detected, rust_percentage
    except Exception as e:
        st.error(f"Error detecting rust: {e}")
        return False, 0

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return 0


def feature_extraction(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None


def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


# File upload section
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Uploaded Image", use_column_width=True)

        # Rust detection
        rust_detected, rust_percentage = detect_rust(os.path.join("uploads", uploaded_file.name))
        if rust_detected:
            st.error(f"Rust detected in the uploaded image! Rust percentage: {rust_percentage:.2f}%")
        else:
            st.success("No rust detected in the uploaded image.")

        # Feature extraction
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        if features is not None:
            st.text("Features: {}".format(features))

            # Recommendation
            indices = recommend(features, feature_list)

            # Display recommendations
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.image(filenames[indices[0][0]], caption=f"Recommendation 1")
            with col2:
                st.image(filenames[indices[0][1]], caption=f"Recommendation 2")
            with col3:
                st.image(filenames[indices[0][2]], caption=f"Recommendation 3")
            with col4:
                st.image(filenames[indices[0][3]], caption=f"Recommendation 4")
            with col5:
                st.image(filenames[indices[0][4]], caption=f"Recommendation 5")
    else:
        st.error("Some error occurred in file upload")
else:
    st.info("Please upload an image.")