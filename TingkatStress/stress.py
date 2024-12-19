# Import modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Preprocessing Data
# Load images and split into training and testing sets
def read_data():
    df = pd.read_csv('TingkatStress\Stress-Lysis.csv')
    return df

def prepare_data():
    df = read_data()
    X = df.iloc[:, :-1].values  # Features
    y = df['Stress_Level'].values       # Labels
    return X, y

def random_forest(X_train, y_train, X_test):
    # Classification with Random Forest
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    return rf.predict(X_test)

# Function to display confusion matrix
def display_confusion_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    # Load data
    X, y = prepare_data()

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Classification with Random Forest
    y_pred_rf = random_forest(X_train, y_train, X_test)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf))
    display_confusion_matrix(y_test, y_pred_rf, "Random Forest Confusion Matrix")