import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pandas as pd

class PredictionPipeline:
    def __init__(self,model):
        self.model = joblib.load(Path(f'artifacts/model_trainer/{model}.joblib'))
        self.scaler = StandardScaler().fit(pd.read_csv(Path(f'artifacts/data_transformation/full_transformed_data.csv')).iloc[:, :-1])  

    def predict(self, data:StandardScaler):
        """
        This function predcit feature based on training.
        arg: data (should be scaled data)
        """
        prediction = self.model.predict(data)

        return prediction