import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt

def load_model(F:\Demo_Class\converted_keras\keras_model.h5):
    """
    Load the pre-trained Keras model
    """
    try:
        model = tf.keras.models.load_model(F:\Demo_Class\converted_keras\keras_model.h5)
        print(f"Successfully loaded model from {F:\Demo_Class\converted_keras\keras_model.h5}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess a single image for prediction
    """
    try:
        # Load and resize image
        img = Image.open(image_path)
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        
        # Handle different image channels
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None, None

def predict_and_display(model, image_path, class_names):
    """
    Make prediction on an image and display the result
    """
    # Preprocess image
    img_array, original_img = preprocess_image(image_path)
    if img_array is None:
        return
    
    try:
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        
        # Display image with prediction
        plt.figure(figsize=(8, 6))
        plt.imshow(original_img)
        plt.title(f'Predicted: {predicted_class}\nConfidence: {confidence:.2%}')
        plt.axis('off')
        plt.show()
        
        return predicted_class, confidence
    except Exception as e:
        print(f"Error making prediction for {image_path}: {e}")
        return None, None

def main():
    # Configuration
    MODEL_PATH = r'F:\Demo_Class\converted_keras\keras_model.h5'  # Using raw string for Windows path
    DATA_DIR = 'Data'
    CLASS_NAMES = ['apple', 'banana', 'tomato', 'carrot']
    
    # Load model
    model = load_model(MODEL_PATH)
    if model is None:
        return
    
    # Get list of image files
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(DATA_DIR, ext)))
    
    if not image_files:
        print(f"No image files found in {DATA_DIR}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Process each image
    results = []
    for image_path in image_files:
        print(f"\nProcessing: {os.path.basename(image_path)}")
        predicted_class, confidence = predict_and_display(model, image_path, CLASS_NAMES)
        if predicted_class is not None:
            results.append({
                'image': os.path.basename(image_path),
                'predicted_class': predicted_class,
                'confidence': confidence
            })
    
    # Print summary
    if results:
        print("\nPrediction Summary:")
        print("-" * 50)
        for result in results:
            print(f"Image: {result['image']}")
            print(f"Predicted Class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print("-" * 50)

if __name__ == "__main__":
    main() 