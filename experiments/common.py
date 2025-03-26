from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import numpy as np
import pandas as pd

def evaluate(y_test, y_pred, label_encoder, heading='-----Evaluation-----'):
    print(heading)
    print('-----Evaluation-----')

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    wandb.log({"Accuracy Score": accuracy})

    # Macro-F1
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Macro F1-score: {macro_f1:.4f}")
    wandb.log({"Macro F1-score": macro_f1})

    # Micro-F1
    micro_f1 = f1_score(y_test, y_pred, average="micro")
    print(f"Micro F1-score: {micro_f1:.4f}")
    wandb.log({"Micro F1-score": micro_f1})

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    categories = np.unique(np.concatenate((y_test, y_pred)))
    df_cm = pd.DataFrame(cm, index=categories, columns=categories)
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='d')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    wandb.log({"Confusion Matrix": wandb.Table(dataframe=df_cm)})