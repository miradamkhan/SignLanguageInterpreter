import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def create_model(input_shape=(64, 64, 1), num_classes=28):
    """
    Create a CNN model for ASL sign recognition
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_image(image_path):
    """
    Preprocess a single image for the model
    """
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Apply thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Resize
    resized = cv2.resize(thresh, (64, 64))
    
    # Normalize
    normalized = resized / 255.0
    
    return normalized

def load_data_from_directory(data_dir, test_split=0.2, val_split=0.1):
    """
    Load and preprocess images from a directory structure
    Directory structure should be:
    data_dir/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            img1.jpg
            ...
        ...
    """
    images = []
    labels = []
    label_map = {}
    
    # Get all subdirectories (classes)
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Create label mapping
    for i, class_name in enumerate(sorted(class_dirs)):
        label_map[class_name] = i
    
    # Load images
    for class_name in class_dirs:
        class_dir = os.path.join(data_dir, class_name)
        class_label = label_map[class_name]
        
        for img_file in os.listdir(class_dir):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, img_file)
                try:
                    # Preprocess the image
                    img_processed = preprocess_image(img_path)
                    
                    # Add to dataset
                    images.append(img_processed)
                    labels.append(class_label)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    # Reshape for the model
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    # Split into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_split/(1-test_split), random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_map

def train_model(model, X_train, X_val, y_train, y_val, batch_size=32, epochs=20, save_path="asl_model.h5"):
    """
    Train the model with the given data
    """
    # Create data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    
    # Create callbacks
    checkpoint = ModelCheckpoint(
        save_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        verbose=1,
        mode='max'
    )
    
    # Train the model
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping]
    )
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data
    """
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    return test_loss, test_acc

def plot_training_history(history):
    """
    Plot the training history
    """
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def main():
    """
    Main function to demonstrate training
    """
    print("ASL Sign Recognition Model Trainer")
    print("==================================")
    
    # Check if dataset path is provided
    import sys
    if len(sys.argv) < 2:
        print("Usage: python model_trainer.py <path_to_dataset>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    print(f"Loading data from {data_dir}...")
    
    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, label_map = load_data_from_directory(data_dir)
    
    print(f"Loaded {len(X_train)} training samples, {len(X_val)} validation samples, {len(X_test)} test samples")
    print(f"Number of classes: {len(label_map)}")
    
    # Create and train model
    model = create_model(input_shape=(64, 64, 1), num_classes=len(label_map))
    model.summary()
    
    print("\nStarting model training...")
    model, history = train_model(model, X_train, X_val, y_train, y_val, epochs=30)
    
    # Evaluate model
    print("\nEvaluating model on test data...")
    evaluate_model(model, X_test, y_test)
    
    # Plot training history
    plot_training_history(history)
    
    # Save label mapping
    import json
    with open('label_map.json', 'w') as f:
        json.dump({v: k for k, v in label_map.items()}, f)
    
    print(f"\nModel saved to asl_model.h5")
    print(f"Label mapping saved to label_map.json")

if __name__ == "__main__":
    main() 