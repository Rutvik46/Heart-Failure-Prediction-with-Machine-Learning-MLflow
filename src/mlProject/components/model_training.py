from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import joblib
from mlProject import logger
from mlProject.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):

        train_data =  pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        scaler = StandardScaler()

        X_train = train_data.drop([self.config.target_column], axis=1)
        X_train = scaler.fit_transform(X_train)
        

        X_test = test_data.drop([self.config.target_column], axis=1)
        X_test = scaler.fit_transform(X_test)

        y_train = train_data[self.config.target_column]
        y_test = test_data[self.config.target_column]

        model = SVC(
            gamma=self.config.gamma,
            C=self.config.C,
            random_state=42 
        )

        model.fit(X_train, y_train)

        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))