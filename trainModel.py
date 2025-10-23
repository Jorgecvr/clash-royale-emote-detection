# train_model.py
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    X, y = [], []
    label_names = sorted(os.listdir(DATA_DIR))
    for label in label_names:
        folder = os.path.join(DATA_DIR, label)
        if not os.path.isdir(folder): continue
        files = [f for f in os.listdir(folder) if f.endswith(".npy")]
        for f in files:
            vec = np.load(os.path.join(folder, f))
            X.append(vec)
            y.append(label)
    return np.array(X), np.array(y), label_names

def main():
    X, y, labels = load_data()
    print("Loaded:", X.shape, "labels:", np.unique(y))
    # shuffle & split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    # Option 1: SVM
    clf = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
    # Optionally run GridSearch (slower). We'll do a simple fit to speed things up.
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    # save model and label order
    joblib.dump({"model": clf, "labels": labels}, os.path.join(MODEL_DIR, "face_expr_model.joblib"))
    print("Model saved to models/face_expr_model.joblib")

if __name__ == "__main__":
    main()
