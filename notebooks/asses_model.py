from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

y_pred = pd.read_csv("wyniki_klasyfikacji.csv")
y_test = pd.read_csv("test_wyniki.csv")

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"F1 score: {f1_score(y_test, y_pred, average='micro')}")