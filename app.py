import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import keras
import os

# Custom layer to handle DepthwiseConv2D compatibility
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

def load_model():
    """Load the pre-trained Keras model"""
    try:
        custom_objects = {
            'DepthwiseConv2D': CustomDepthwiseConv2D
        }
        model = keras.models.load_model('converted_keras/keras_model.h5', custom_objects=custom_objects, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess the uploaded image for prediction"""
    try:
        # Resize image
        image = image.resize(target_size)
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        
        # Handle different image channels
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def main():
    # Set page config
    st.set_page_config(
        page_title="Fruit & Vegetable Classifier",
        page_icon="🥗",
        layout="centered"
    )
    
    # Title and description
    st.title("🍎 Fruit & Vegetable Classifier")
    st.markdown("""
    Upload an image of an apple, banana, tomato, or carrot to classify it.
    The model will predict which category the image belongs to.
    """)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load the model. Please check if 'keras_model.h5' exists in the correct location.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image of an apple, banana, tomato, or carrot"
    )
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess image
            img_array = preprocess_image(image)
            if img_array is None:
                return
            
            # Make prediction with loading spinner
            with st.spinner("Analyzing image..."):
                predictions = model.predict(img_array)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                
                # Class names
                class_names = ['apple', 'banana', 'carrot', 'tomato']
                predicted_class = class_names[predicted_class_idx]
                
                # Display results
                st.success(f"Prediction: {predicted_class.capitalize()}")
                st.info(f"Confidence: {confidence:.2%}")
                
                # Display confidence scores for all classes
                st.subheader("Confidence Scores:")
                for i, class_name in enumerate(class_names):
                    st.metric(
                        label=class_name.capitalize(),
                        value=f"{predictions[0][i]:.2%}"
                    )
                
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main() 