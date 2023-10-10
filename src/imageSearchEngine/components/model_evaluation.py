import os
from tqdm import tqdm
import pickle
from typing import List
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow .keras.applications import resnet50
import numpy as np
from imageSearchEngine.logging.logger import log
from imageSearchEngine.exception import CustomException
from imageSearchEngine.config.configuration import ModelEvaluationConfig

class ModelEvaluation:

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    @staticmethod
    def _precision(predicted, actual):
        prec = [value for value in predicted if value in actual]
        prec = float(len(prec)) / float(len(predicted))
        return prec
    
    @staticmethod
    def _apk(actual: list, predicted: list, k=10) -> float:
        """
        Computes the average precision at k.
        Parameters
        ----------
        actual : list
            A list of actual items to be predicted
        predicted : list
            An ordered list of predicted items
        k : int, default = 10
            Number of predictions to consider
        Returns:
        -------
        score : float
            The average precision at k.
        """
        if not predicted or not actual:
            return 0.0
        
        if len(predicted) > k:
            predicted = predicted[:k]

        score = 0.0
        true_positives = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                max_ix = min(i + 1, len(predicted))
                score += ModelEvaluation._precision(predicted[:max_ix], actual)
                true_positives += 1
        
        if score == 0.0:
            return 0.0
        
        return score / true_positives
    
    @staticmethod
    def mapk(actual: List[list], predicted: List[list], k: int=10) -> float:
        """
        Computes the mean average precision at k.
        Parameters
        ----------
        actual : a list of lists
            Actual items to be predicted
            example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
        predicted : a list of lists
            Ordered predictions
            example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
        Returns:
        -------
            mark: float
                The mean average precision at k (map@k)
        """
        if len(actual) != len(predicted):
            raise AssertionError("Length mismatched")
        
        return np.mean([ModelEvaluation._apk(a,p,k) for a,p in zip(actual, predicted)])

    def create_val_features(self):
        val_features = []
        val_image_path = []
        for label in os.listdir(self.config.val_path):
            image_path = os.path.join(self.config.val_path, f'{label}/images')
            for image in os.listdir(image_path):
                val_image_path.append(os.path.join(image_path, image))
        log.info('all the val_path were retrived')

        model = resnet50.ResNet50(
            include_top=self.config.include_top,
            input_shape=self.config.input_shape,
            pooling=self.config.pooling
        )
        log.info('got resnet50 model from tensorflow')
        log.info("new logs")
        for image_path in tqdm(val_image_path):
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            preprocess_image = resnet50.preprocess_input(image)
            feature = model.predict(preprocess_image, verbose=0).flatten()
            # getting label for the image
            label = image_path.split('/')[-3]
            log.info(label)
            val_features.append((feature, label))
        
        log.info('All the validation data was loaded')
        pickle.dump(val_features, open(self.config.val_feature, 'wb'))
        log.info('All the validation data was saved to thier repsective path')
    
    def get_predictions(self):
        model = pickle.load(open(self.config.model_path, 'rb'))
        predictions = []
        test_labels = []
        #make sure we have the features
        if not os.path.exists(self.config.val_feature):
            log.info("the path for the label feature does not exist creating them now")
            self.create_val_features()
        
        test_feature = pickle.load(open(self.config.val_feature, 'rb'))
        predictions = []
        test_labels = []
        labels = pickle.load(open(self.config.image_labels_path, 'rb'))
        image_path_list = pickle.load(open(self.config.image_path_list_dir, 'rb'))
        model: KNeighborsClassifier = pickle.load(open(self.config.model_path, 'rb'))

        for feature_x, label in tqdm(test_feature):
            index_list = model.kneighbors(feature_x.reshape(1, -1),
                                          n_neighbors=self.config.n_neighbors, return_distance=self.config.return_distance
                                          )[0]
            prediction_name = [image_path_list[i].split('/')[-1] for i in  index_list]
            predictions.append(prediction_name)
            
            test_labels.append(labels[label])
        log.info("all the predictions were made for the test dataset")
        
        return predictions, test_labels

    def evaluate_model(self):
        '''
        find the evaluation score of the model
        '''
        predictions, test_labels = self.get_predictions()
        score = self.mapk(test_labels, predicted=predictions, k=self.config.n_neighbors)
        log.info(f"the score is for the model is {score}")
        
        
        


