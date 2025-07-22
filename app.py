from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route("/train", methods=['POST'])
def train():
    try:
        os.system("python main.py")
        return jsonify({"message": "Training completed successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Input form features in order
            features = [
                'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
                'Oldpeak', 'ST_Slope'
            ]

            # Read model name
            model = request.form.get('model')

            # Read input values from form
            input_data = {}
            for feature in features:
                value = request.form.get(feature)
                if value is None:
                    raise ValueError(f"Missing input for {feature}")
                input_data[feature] = value

            # Convert types
            numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
            for feature in numeric_features:
                input_data[feature] = float(input_data[feature]) if feature == 'Oldpeak' else int(input_data[feature])

            # Create DataFrame from input
            data = pd.DataFrame([input_data])

            # Map categorical values to numeric as required
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

            print(data)

            # Ensure feature order is preserved
            data = data[features]
            
            # Predict
            pipeline = PredictionPipeline(model)
            scaled_data = pipeline.scaler.transform(data)
            scaled_data_df = pd.DataFrame(scaled_data, columns=features)
            prediction = pipeline.predict(scaled_data_df)

            return render_template('results.html', prediction=" Possibility of Heart Failure is High" if str(prediction[0]=='1') else "No Possibility of Heart Failure")

        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return render_template('predict.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
