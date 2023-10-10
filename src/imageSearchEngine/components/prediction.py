import os
from tqdm import tqdm
import pickle
from PIL import Image
from typing import List
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow .keras.applications import resnet50
from tensorflow.image import resize
import numpy as np
from imageSearchEngine.exception import CustomException
from imageSearchEngine.config.configuration import PredictionConfig

class Prediction:

    def __init__(self, config: PredictionConfig):
        self.config = config


    def extract(self, image):
        model = resnet50.ResNet50(
            include_top=self.config.include_top,
            input_shape=self.config.input_shape,
            pooling=self.config.pooling
        )
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = resize(image, self.config.target_size)
        preprocess_image = resnet50.preprocess_input(image)
        feature = model.predict(preprocess_image, verbose=0).flatten()
        return feature
        

    
    def predict(self, image) -> List:
        im = Image.open(image)
        feature = self.extract(im)
        image_path_list = pickle.load(open(self.config.image_path_list_dir, 'rb'))
        model: KNeighborsClassifier = pickle.load(open(self.config.model_path, 'rb'))
        index_list = model.kneighbors(feature.reshape(1, -1),
                                        n_neighbors=self.config.n_neighbors, return_distance=self.config.return_distance
                                        )[0]
        prediction_path = [image_path_list[i] for i in  index_list]
        return prediction_path

