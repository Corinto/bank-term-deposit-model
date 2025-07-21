import joblib
import pandas as pd

DATA_INFER_SOURCE_FILE_NAME = 'bank.csv'
DATA_INFER_SOURCE_PATH = "../data/bank/"

MODEL_PATH = "saved_models/"
MODEL_NAME = "model"

PREDICTIONS_FILE_PATH = "./predictions/"
PREDICTIONS_FILE_NAME = "bank_model_predictions.csv"

# Load dataset for inference
data = pd.read_csv(DATA_INFER_SOURCE_PATH + DATA_INFER_SOURCE_FILE_NAME, delimiter = ';')

# Features used
features = [
  'age',
  'balance',
  'day',
  'duration',
  'campaign',
  'pdays',
  'previous',
  'job',
  'marital',
  'education',
  'default',
  'housing',
  'loan',
  'contact',
  'month',
  'poutcome',
]

X_infer_test = data[features]

# Load the model from disk
model = joblib.load(MODEL_PATH + MODEL_NAME + ".joblib")

# Inference - note that preprocessing is already included in saved pipeline
Y_infer_pred = model.predict(X_infer_test)
data["y_model_prediction"] = Y_infer_pred

# Add also probability of positive class
Y_infer_pred_proba = model.predict_proba(X_infer_test)
Y_infer_pred_proba = Y_infer_pred_proba[:, 1]
data["y_model_proba"] = pd.Series(Y_infer_pred_proba, index = data.index)

# Save predictions in new file
data.to_csv(PREDICTIONS_FILE_PATH + PREDICTIONS_FILE_NAME)

print("Inference done using as source file: ", DATA_INFER_SOURCE_FILE_NAME)
print("Model predictions saved to: ", PREDICTIONS_FILE_PATH + PREDICTIONS_FILE_NAME)


