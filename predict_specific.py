import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import keras

class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        # Remove the 'groups' parameter if it exists
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

def load_model(model_path):
    """
    Load the pre-trained Keras model
    """
    try:
        # Register custom layer
        custom_objects = {
            'DepthwiseConv2D': CustomDepthwiseConv2D
        }
        model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        print(f"Successfully loaded model from {model_path}")
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
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
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
        return None, None
    
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
    MODEL_PATH = r'F:\Demo_Class\converted_keras\keras_model.h5'
    CLASS_NAMES = ['apple', 'banana', 'carrot', 'tomato']
    
    # List of specific images to process
    image_paths = [
        # Apple images
        r'F:\Demo_Class\Data\Apple(5-10)\frame164.jpg',
        r'F:\Demo_Class\Data\Apple(5-10)\frame179.jpg',
        r'F:\Demo_Class\Data\Apple(10-14)\frame37.jpg',
        
        # Banana images
        r'F:\Demo_Class\Data\Banana(5-10)\frame285.jpg',
        r'F:\Demo_Class\Data\Banana(5-10)\frame1880.jpg',
        r'F:\Demo_Class\Data\Banana(15-20)\frame9.jpg',
        
        # Carrot images
        r'F:\Demo_Class\Data\carrot(5-6)\frame20.jpg',
        r'F:\Demo_Class\Data\carrot(5-6)\frame100.jpg',
        r'F:\Demo_Class\Data\Carrot(1-2)\frame190.jpg',
        
        # Tomato images
        r'F:\Demo_Class\Data\Tomato(1-5)\frame180.jpg',
        r'F:\Demo_Class\Data\Tomato(1-5)\frame570.jpg',
        r'F:\Demo_Class\Data\Tomato(1-5)\frame740.jpg'
    ]
    
    # Load model
    model = load_model(MODEL_PATH)
    if model is None:
        return
    
    # Process each image
    results = []
    for image_path in image_paths:
        print(f"\nProcessing: {image_path}")
        if not os.path.exists(image_path):
            print(f"Error: File not found - {image_path}")
            continue
            
        predicted_class, confidence = predict_and_display(model, image_path, CLASS_NAMES)
        if predicted_class is not None:
            results.append({
                'image': image_path,
                'predicted_class': predicted_class,
                'confidence': confidence
            })
    
    # Print summary
    if results:
        print("\nPrediction Summary:")
        print("-" * 80)
        for result in results:
            print(f"Image: {result['image']}")
            print(f"Predicted Class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print("-" * 80)

if __name__ == "__main__":
    main() 