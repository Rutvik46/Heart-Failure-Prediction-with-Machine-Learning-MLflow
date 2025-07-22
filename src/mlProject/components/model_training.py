from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble  import AdaBoostClassifier 
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

        SVM = SVC(
            gamma=self.config.model_param.Support_Vector_Machine.gamma,
            C=self.config.model_param.Support_Vector_Machine.C,
            kernel=self.config.model_param.Support_Vector_Machine.kernel,
            random_state=42 
        )

        # SVM
        SVM.fit(X_train, y_train)
        joblib.dump(SVM, os.path.join(self.config.root_dir, self.config.model_names.model1))

        # KNN
        KNN=KNeighborsClassifier(n_neighbors=self.config.model_param.Knearest_Neighbour.n_neighbours)
        KNN.fit(X_train, y_train)
        joblib.dump(KNN, os.path.join(self.config.root_dir, self.config.model_names.model2))

        # AdaBoostClassifier
        AdaBoost= AdaBoostClassifier(n_estimators=self.config.model_param.AdaBoost.n_estimators,
                                   random_state=42
        )
        AdaBoost.fit(X_train, y_train)
        joblib.dump(AdaBoost, os.path.join(self.config.root_dir, self.config.model_names.model3))