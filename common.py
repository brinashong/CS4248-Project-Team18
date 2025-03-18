from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate(y_val, y_pred, label_encoder, heading='-----Evaluation-----'):
    print(heading)
    print('-----Evaluation-----')
    
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    print(f"Macro F1-score: {macro_f1:.4f}")
    
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))
    
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()