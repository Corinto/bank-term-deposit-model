import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline


DATA_SOURCE_FILE_NAME = 'bank-full.csv'
DATA_SOURCE_PATH = "../data/bank/"

MODEL_PATH = "saved_models/"
MODEL_NAME = "model"

# Load dataset
print("Loading dataset:")
data = pd.read_csv( DATA_SOURCE_PATH + DATA_SOURCE_FILE_NAME, delimiter = ';')
print("OK.")

# Select model features
numerical_features = [
  'age',
  'balance',
  'day',
  'duration',
  'campaign',
  'pdays',
  'previous',
]

categorical_features = [
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

# Target variable to predict
target = 'y'

# Train - test split
X_train, X_test, Y_train, Y_test = train_test_split(data.iloc[:,:-1], 
                                                    data.iloc[:,-1], 
                                                    test_size=0.3, 
                                                    random_state=43)

XY_train = pd.concat([X_train, Y_train], axis = 1)
XY_test = pd.concat([X_test, Y_test], axis = 1)

# ML pipeline - construct the preprocessing pipeline

cat_var = [X_train.columns.get_loc(col) for col in categorical_features]

# Note dataset has no missing values but we add imputers in case inference dataset contains
# missing values 
numeric_preprocessing = Pipeline(steps=[("imputer", SimpleImputer(strategy='median')),
                                        ("scaler", StandardScaler())])

cat_preprocessing = Pipeline(steps=[("imputer", SimpleImputer(strategy='most_frequent')),
                                    ("encoder", OneHotEncoder(handle_unknown="ignore"))])

preprocessing = ColumnTransformer([("numeric", numeric_preprocessing, numerical_features),
                                   ("cat", cat_preprocessing, cat_var)])

# Model selection - RF pipeline
rf_pipeline = Pipeline([("preprocessing", preprocessing),
                        ("rf_classifier", RandomForestClassifier(random_state=0))])

rf_parameters = {'rf_classifier__n_estimators':[32, 64, 100],
                 'rf_classifier__min_samples_split':[2,5,10,20]}

rf_model = GridSearchCV(rf_pipeline, param_grid=rf_parameters, scoring='roc_auc', cv=3)

# time 
print("Performing grid search to find best model: ~5 mins")
rf_model.fit(X_train, Y_train)

print(pd.DataFrame(rf_model.cv_results_))

print("Best model found: ", rf_model.best_params_)

# Save model
print("Saving model")
joblib.dump(rf_model.best_estimator_, MODEL_PATH + MODEL_NAME + ".joblib", compress = 1)

# Load the model from disk
model = joblib.load(MODEL_PATH + MODEL_NAME + ".joblib")

# Display evaluation metrics on test set
Y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

# Compute AUROC curve and save result
Y_test_b = [1 if x == 'yes' else 0 for x in Y_test]
Y_pred_proba = model.predict_proba(X_test)
Y_pred_proba = Y_pred_proba[:, 1]
fpr, tpr, thresholds = roc_curve(Y_test_b, Y_pred_proba)
roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
display.plot()
plt.savefig(MODEL_PATH + MODEL_NAME + '_auc.png')
plt.show()

print("Best model saved to: ", MODEL_PATH)



