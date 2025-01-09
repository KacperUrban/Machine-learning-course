from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import os
import numpy as np

y_pred = pd.read_csv(os.path.join("results", "wyniki_klasyfikacji.csv"))
y_test = pd.read_csv(os.path.join("data", "test_pokemon.csv"))

print(f"Accuracy: {np.round(accuracy_score(y_test, y_pred), 4)}")
print(f"F1 score: {np.round(f1_score(y_test, y_pred, average='micro'), 4)}")