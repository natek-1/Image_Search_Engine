import os
from tqdm import tqdm
from pathlib import Path
import pickle
from typing import List
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from imageSearchEngine.logging.logger import log
from imageSearchEngine.config.configuration import ModelTrainerConfig


class ModelTrainer:

    def __init__(self, config:ModelTrainerConfig):
        self.config = config
    
    def get_image_class(self) -> List:
        '''
        find the image clases based on the path
        return List[str]
        '''
        path_list = pickle.load(open(self.config.image_path_list_dir,'rb'))
        image_class = []
        for image_path in tqdm(path_list):
            class_name = image_path.split('/')[-3]
            image_class.append(class_name)
        log.info(f"retrived all the image classes from the path: {self.config.image_path_list_dir}")
        return image_class
    
    def train_model(self):
        '''
        trains the mode and saves to the path according to path in config file
        '''
        image_labels = self.get_image_class()
        image_feature = pickle.load(open(self.config.feature_dir,'rb'))
        log.info(f"retrived all the features images that were produced earlier in the path: {self.config.feature_dir}")
        X = np.array(image_feature)
        y = np.array(image_labels)
        knn_classifier = KNeighborsClassifier(n_jobs=-1, n_neighbors=self.config.n_neighbors)
        knn_classifier = knn_classifier.fit(X, y)
        log.info("the model was fully trained")
        
        with open(self.config.model_path, 'wb') as file:
            pickle.dump(knn_classifier, file) 
        log.info(f"this model was saved to the path, {self.config.model_path}")