import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from concurrent.futures import ThreadPoolExecutor
model =ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
#print(model.summary())

def extract_features(img_paths, model):
    img_list = [image.load_img(path, target_size=(224, 224)) for path in img_paths]
    img_array_list = [image.img_to_array(img) for img in img_list]
    expanded_img_array_list = [np.expand_dims(img_array, axis=0) for img_array in img_array_list]
    preprocessed_imgs = preprocess_input(np.concatenate(expanded_img_array_list, axis=0))
    results = model.predict(preprocessed_imgs)
    normalized_results = [result / norm(result) for result in results]
    
    return normalized_results


# Directory containing images
image_directory = 'images'

# Get list of image file paths
filenames = [os.path.join(image_directory, file) for file in os.listdir(image_directory)]

# Batch size for processing
batch_size = 32

# Initialize feature list
feature_list = []

# Function to process a batch of files and extend the feature list
def process_batch(batch_files):
    return extract_features(batch_files, model)

# Process images using ThreadPoolExecutor
with ThreadPoolExecutor() as executor:
    for i in tqdm(range(0, len(filenames), batch_size)):
        batch_files = filenames[i:i + batch_size]
        feature_list.extend(executor.map(process_batch, [batch_files]))

# Flatten the list of features
feature_list = [feature for sublist in feature_list for feature in sublist]

# Save the features and filenames
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb')) 