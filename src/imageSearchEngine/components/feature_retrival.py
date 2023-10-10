import os
from tqdm import tqdm
import pickle
from typing import List
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow .keras.applications import resnet50
import numpy as np
from imageSearchEngine.logging.logger import log
from imageSearchEngine.exception import CustomException
from imageSearchEngine.config.configuration import FeatureRetrivalConfig


class FeatureRetrival:

    def __init__(self, config: FeatureRetrivalConfig):
        self.config = config
    
    def retrive_labels(self):
        labels = {}
        for label in os.listdir(self.config.data_path):
            image_path = os.path.join(self.config.data_path, f'{label}/images')
            labels[label] = os.listdir(image_path)
        log.info('all the labels and images were retrived')
        pickle.dump(labels, open(self.config.image_labels, 'wb'))
        log.info(f'all the labels were saved to {self.config.image_labels}')
        

    def retrive_path(self) -> List:
        image_path_list = []
        for label in os.listdir(self.config.data_path):
            image_path = os.path.join(self.config.data_path, f'{label}/images')
            for image in os.listdir(image_path):
                image_path_list.append(os.path.join(image_path, image))
        log.info('all the image_path were retrived')
        with open(self.config.image_path_list_dir, 'wb') as file:
            pickle.dump(image_path_list, file)
        log.info('all image paths were saved')
        return image_path_list
    
    def retrive_embedding(self):
        model = resnet50.ResNet50(
            include_top=self.config.include_top,
            input_shape=self.config.input_shape,
            pooling=self.config.pooling,
            weights='imagenet',
        )
        log.info('got resnet50 model from tensorflow')
        image_path_list = self.retrive_path()
        image_feature = []
        for image_path in tqdm(image_path_list):
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            preprocess_image = resnet50.preprocess_input(image)
            feature = model.predict(preprocess_image, verbose=0).flatten()
            image_feature.append(feature)
        log.info('all the features were loaded')
        pickle.dump(image_feature, open(self.config.feature_dir, 'wb'))
        log.info('all the feauture representaion where saved')
        
