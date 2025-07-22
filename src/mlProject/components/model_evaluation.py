import pandas as pd
from mlProject import logger
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from urllib.parse import urlparse
import mlflow
import numpy as np
import joblib
from mlProject.utils.common import save_json
from mlProject.entity.config_entity import ModelEvaluationConfig
from pathlib import Path
import time 

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluation_metrics(self,actual,pred):
        Accuracy = accuracy_score(actual,pred)
        F1_Score = f1_score(actual,pred)
        Precision_Score = precision_score(actual,pred)
        Recall_Score = recall_score(actual,pred)
        AUC_Score = roc_auc_score(actual, pred)
        
        return Accuracy, F1_Score, Precision_Score, Recall_Score, AUC_Score

    def log_into_mlflow(self):

        # Load models
        SVM = joblib.load(Path(self.config.model_paths[0]))
        KNN = joblib.load(Path(self.config.model_paths[1]))
        AdaBoost = joblib.load(Path(self.config.model_paths[2]))

        models = {"Support_Vector_Machine":SVM,"Knearest_Neighbour":KNN, "AdaBoost":AdaBoost}

        # Load test data
        test_data = pd.read_csv(self.config.test_data_path)
        X_test = test_data.drop([self.config.target_column], axis=1)
        Y_test = test_data[self.config.target_column]

        for index, model_name in enumerate(models.keys()):
            mlflow.set_registry_uri(self.config.mlflow_url)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():

                predicted_qualities = models[model_name].predict(X_test)
        
                # Evaluation metrcis
                (Accuracy, F1_Score, Precision_Score, Recall_Score, AUC_Score) = self.evaluation_metrics(Y_test, predicted_qualities)
                
                # Saving metrics as local
                scores = {"Accuracy":Accuracy, "F1_Score":F1_Score, "Precision_Score":Precision_Score, "Recall_Score":Recall_Score, "AUC_Score":AUC_Score}
                save_json(path=Path(self.config.metrics_file_paths[index]), data=scores)

                '''
                mlflow.log_params(self.config.all_params[model_name])

                mlflow.log_metric("Accuracy",Accuracy)
                mlflow.log_metric("F1_Score",F1_Score)
                mlflow.log_metric("Precision_Score",Precision_Score)
                mlflow.log_metric("Recall_Score",Recall_Score)
                mlflow.log_metric("AUC_Score",AUC_Score)


                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(models[model_name], "model", registered_model_name=model_name)
                else:
                    mlflow.sklearn.log_model(models[model_name], "model")'''