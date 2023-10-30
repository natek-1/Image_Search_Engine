import os
import streamlit as st 
import pickle
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow .keras.applications import resnet50
from tensorflow.image import resize
import cv2
from PIL import Image
import numpy as np
from imageSearchEngine.utils.file_helpers import create_directories

UPLOAD_PATH= "artifacts/uploads"

# save the uploaded image
def save_upload_image(upload_image, upload_path=UPLOAD_PATH):
    try:
        os.makedirs(upload_path, exist_ok=True)
        with open(os.path.join(upload_path, upload_image.name), 'wb') as image:
            image.write(upload_image.getbuffer())
        return True
    except Exception as e:
        return False

include_top= False
pooling= "avg"
input_shape= [224, 224, 3]
target_size= [224, 224]

extractor = resnet50.ResNet50(
            include_top=include_top,
            input_shape=input_shape,
            pooling=pooling
        )


def extract_features(img_path,extract=extractor):
    image = Image.open(img_path)
    image = resize(image, target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = np.copy(image)
    image = image[:,:,:,:3]
    preprocess_image = resnet50.preprocess_input(image)
    feature = extractor.predict(preprocess_image, verbose=0).flatten()
    return feature


MODEL_PATH = "artifacts/model/image_search_engine_classifier.pkl"
model = pickle.load(open(MODEL_PATH, 'rb'))
image_path_list_dir = "artifacts/embeddings/path_list.pkl"
image_path_list = pickle.load(open(image_path_list_dir, 'rb'))


def predict(feature, model=model):
    predictions = []
    index_list = model.kneighbors(feature.reshape(1, -1),
                                n_neighbors=20, return_distance=False
                                )[0]
    prediction_name = [image_path_list[i] for i in  index_list]
    predictions.append(prediction_name)
    return predictions[0]


st.title("Image search engine")



upload_image = st.file_uploader(label="Upload image of what you want to search")
if upload_image is not None:
    display_image = st.image(upload_image)
    if save_upload_image(upload_image):
        feature = extract_features(os.path.join(UPLOAD_PATH, upload_image.name))
        predictions = predict(feature)
        st.subheader("Similar Images")
        st.image(predictions, width=150)


    

