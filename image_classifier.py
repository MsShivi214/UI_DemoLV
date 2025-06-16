import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import glob

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def load_and_preprocess_data(data_dir, img_size=(224, 224)):
    """
    Load and preprocess images from the specified directory.
    Expected directory structure:
    data_dir/
        apple/
        banana/
        tomato/
        carrot/
    """
    images = []
    labels = []
    class_names = ['apple', 'banana', 'tomato', 'carrot']
    
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        image_files = glob.glob(os.path.join(class_path, '*.*'))
        
        for img_path in image_files:
            try:
                # Load and preprocess image
                img = Image.open(img_path)
                img = img.resize(img_size)
                img = np.array(img) / 255.0  # Normalize to [0, 1]
                
                # Handle different image channels
                if len(img.shape) == 2:  # Grayscale
                    img = np.stack((img,) * 3, axis=-1)
                elif img.shape[2] == 4:  # RGBA
                    img = img[:, :, :3]
                
                images.append(img)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    
    return np.array(images), np.array(labels)

def create_model(input_shape=(224, 224, 3), num_classes=4):
    """
    Create a CNN model suitable for small to medium datasets
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def predict_image(model, image_path, class_names):
    """
    Make prediction on a single image
    """
    # Load and preprocess image
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    
    # Handle different image channels
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return predicted_class, confidence

def main():
    # Configuration
    DATA_DIR = "path/to/your/dataset"  # Update this path
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 20
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data(DATA_DIR, IMG_SIZE)
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Create and compile model
    print("Creating model...")
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val)
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save the model
    model.save('fruit_vegetable_classifier.h5')
    print("Model saved as 'fruit_vegetable_classifier.h5'")

if __name__ == "__main__":
    main() 