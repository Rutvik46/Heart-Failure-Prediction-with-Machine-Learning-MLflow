import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlProject import logger
from mlProject.entity.config_entity import DataTransformationConfig
import os

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        

    # All data transformation techniques such as normalization, encoding, Scalar, PCA, etc. can be implemented here.
    
    def data_preprocessing(self) -> pd.DataFrame:

        logger.info("Starting data preprocessing...")

        # Load the data
        data = pd.read_csv(self.config.data_path)
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)

        # Map categorical columns to numeric
        data['Sex'] = data['Sex'].map({'M': 1, 'F': 0})
        data['ExerciseAngina'] = data['ExerciseAngina'].map({'Y': 1, 'N': 0})


        data['ChestPainType'] = data['ChestPainType'].map({
            'ATA': 0,
            'NAP': 1,
            'ASY': 2,
            'TA': 3
        })

        data['RestingECG'] = data['RestingECG'].map({
            'Normal': 0,
            'ST': 1,
            'LVH': 2
        })

        data['ST_Slope'] = data['ST_Slope'].map({
            'Up': 0,
            'Flat': 1,
            'Down': 2
        })

        # Save mapped data 
        data.to_csv(os.path.join(self.config.root_dir ,'full_transformed_data.csv'), index=False)

        # Standardize the data for feature scaling (to treat each feature equally important)
        features = data.iloc[:, :-1]   # all columns except last
        target = data.iloc[:, -1]

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Combine scaled features and original target
        processed_data = pd.DataFrame(scaled_features, columns=features.columns)
        processed_data[target.name] = target.values  # add target back as last column

        return processed_data

    def train_test_split(self, data: pd.DataFrame) -> None:
        
        # Splitting the data into train and test sets
        # 75% of the data will be used for training and 25% for testing
        train, test = train_test_split(data, test_size=0.25, random_state=42)

        train.to_csv(os.path.join(self.config.root_dir ,'train.csv'), index=False)
        test.to_csv(os.path.join(self.config.root_dir ,'test.csv'), index=False)

        logger.info(f"Train and test data saved at {self.config.root_dir}")
        logger.info(f"Train data shape: {train.shape}")
        logger.info(f"Test data shape: {test.shape}")