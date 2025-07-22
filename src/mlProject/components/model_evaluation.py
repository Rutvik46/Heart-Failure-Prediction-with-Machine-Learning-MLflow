import pandas as pd
from mlProject import logger
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from urllib.parse import urlparse
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from mlProject.utils.common import save_json
from mlProject.entity.config_entity import ModelEvaluationConfig
from pathlib import Path 

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

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        scaler = StandardScaler()

        X_test = test_data.drop([self.config.target_column], axis=1)
        X_test = scaler.fit_transform(X_test)

        Y_test = test_data[self.config.target_column]


        mlflow.set_registry_uri(self.config.mlflow_url)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


        with mlflow.start_run():

            predicted_qualities = model.predict(X_test)

            (Accuracy, F1_Score, Precision_Score, Recall_Score, AUC_Score) = self.evaluation_metrics(Y_test, predicted_qualities)
            
            # Saving metrics as local
            scores = {"Accuracy":Accuracy, "F1_Score":F1_Score, "Precision_Score":Precision_Score, "Recall_Score":Recall_Score, "AUC_Score":AUC_Score}
            save_json(path=Path(self.config.metrics_file_path), data=scores)

            mlflow.log_params(self.config.all_params)

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
                mlflow.sklearn.log_model(model, "model", registered_model_name="Support Vector Machine")
            else:
                mlflow.sklearn.log_model(model, "model")