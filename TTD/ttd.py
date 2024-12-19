import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


def prepare_data(data_dir, img_size=(128, 128), test_split=0.2):
    """Load images, resize them, and split into training and testing sets."""
    data, labels = [], []
    class_names = os.listdir(data_dir)

    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    data.append(img.flatten())
                    labels.append(idx)

    data = np.array(data) / 255.0  # Normalize pixel values to [0, 1]
    labels = np.array(labels)

    return train_test_split(data, labels, test_size=test_split, random_state=42), class_names

def train_model(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, class_names):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=class_names))
    return accuracy

def save_model(model, model_path="signature_model.pkl"):
    """Save the trained model to a file."""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(model_path="signature_model.pkl"):
    """Load a trained model from a file."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def plot_sample_images(X, y, class_names, img_size=(128, 128), num_samples=1):
    """Plot sample images from the dataset."""
    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        idx = np.random.randint(0, len(X))
        img = X[idx].reshape(img_size)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(class_names[y[idx]])
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Define the path to the dataset directly in the code
    data_dir =r'D:\DATA VIA/UASKB/TTD'  # Replace with the path to your dataset

    # Prepare data with 20% reserved for testing
    (X_train, X_test, y_train, y_test), class_names = prepare_data(data_dir, test_split=0.2)
    print(f"Dataset prepared: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # Plot sample images
    plot_sample_images(X_train, y_train, class_names)

    # Train model
    print("Training model...")
    model = train_model(X_train, y_train)

    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test, class_names)

    # Save model
    save_model(model)
    print("Model saved as 'signature_model.pkl'.")
