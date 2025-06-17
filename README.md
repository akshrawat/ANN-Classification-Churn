# Customer Churn Prediction Web App

This project is a web application for predicting customer churn using an Artificial Neural Network (ANN) model. The app allows users to input customer details and returns a prediction on whether the customer is likely to churn.

## Features

- Predicts customer churn based on user input
- Utilizes a pre-trained ANN model (`model.h5`)
- Encodes categorical variables using pre-fitted encoders (`label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`)
- Scales input features using a pre-fitted scaler (`scaler.pkl`)
- User-friendly web interface built with Flask

## Project Structure

```
ANN_model_churning/
├── app.py
├── Churn_Modelling.csv
├── experiments.ipynb
├── label_encoder_gender.pkl
├── model.h5
├── onehot_encoder_geo.pkl
├── prediction.ipynb
├── requirements.txt
├── scaler.pkl
├── logs/
│   └── ...
├── model.tf/
│   └── ...
```

- `app.py`: Main Flask application
- `Churn_Modelling.csv`: Dataset used for training
- `label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`, `scaler.pkl`: Pre-fitted encoders and scaler
- `experiments.ipynb`, `prediction.ipynb`: Notebooks for model development and prediction
- `logs/`: Training logs
- `model.tf/`: TensorFlow SavedModel format

## Setup Instructions

1. **Clone the repository** and navigate to the `ANN_model_churning` directory.

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```
   python app.py
   ```

4. **Open your browser** and go to `http://localhost:5000` to use the app.

## Usage

- Enter customer details in the web form.
- Click "Predict" to see if the customer is likely to churn.

## Requirements

See [`requirements.txt`](ANN_model_churning/requirements.txt) for the full list of dependencies.

## License

This project is for educational purposes.

---

For more details, see [`app.py`](ANN_model_churning/app.py) and the included Jupyter
