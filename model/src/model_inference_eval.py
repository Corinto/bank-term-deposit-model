import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay, auc, roc_curve

# In this case we actually have the labels for the inference dataset 
# We can evaluate how the model is performing in this new unseen dataset

DATA_SOURCE_FILE_NAME = 'bank.csv'
DATA_SOURCE_PATH = "../data/bank/"

PREDICTIONS_FILE_PATH = "predictions/"
PREDICTIONS_FILE_NAME = "bank_model_predictions.csv"

# Load datasets for evaluation
data_source = pd.read_csv(DATA_SOURCE_PATH + DATA_SOURCE_FILE_NAME, delimiter = ';')
Y_true = data_source["y"]

data_predicted = pd.read_csv(PREDICTIONS_FILE_PATH + PREDICTIONS_FILE_NAME)
Y_pred = data_predicted["y_model_prediction"]
Y_pred_proba = data_predicted["y_model_proba"]

print("Accuracy:", accuracy_score(Y_true, Y_pred))
print(classification_report(Y_true, Y_pred))

# Compute AUROC curve
Y_true_bry = [1 if x == 'yes' else 0 for x in Y_true]
fpr, tpr, thresholds = roc_curve(Y_true_bry, Y_pred_proba)
roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
display.plot()
plt.show()