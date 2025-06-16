# UI_DemoLV

```markdown
# Fruit & Vegetable Image Classification Web App

A Streamlit-based web application for classifying images of fruits and vegetables into four categories: apple, banana, tomato, and carrot.

## Features

- User-friendly web interface for image upload
- Real-time image classification
- Support for multiple image formats (JPG, JPEG, PNG)
- Displays prediction confidence scores for all classes
- Handles various image types (RGB, RGBA, Grayscale)
- Loading spinner during prediction
- Error handling for unsupported files and corrupted images

## Prerequisites

- Python 3.12 or higher
- pip (Python package installer)

## Installation

1. Clone the repository or download the source code.

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

The following packages will be installed:
- streamlit==1.32.0
- tensorflow==2.19.0
- keras>=3.5.0
- numpy
- Pillow

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── converted_keras/       # Directory containing the model
│   ├── keras_model.h5    # Pre-trained Keras model
│   └── labels.txt        # Class labels
├── requirements.txt      # Python dependencies
└── Data/                 # Sample images for testing
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload an image using the file uploader:
   - Supported formats: JPG, JPEG, PNG
   - The image should be of an apple, banana, tomato, or carrot

4. View the results:
   - The uploaded image will be displayed
   - The predicted class will be shown with confidence score
   - Confidence scores for all classes will be displayed

## Model Information

- The model is a pre-trained Keras model saved in H5 format
- Input image size: 224x224 pixels
- Output classes: ['apple', 'banana', 'carrot', 'tomato']
- The model automatically handles image preprocessing:
  - Resizing to 224x224
  - Normalization (pixel values scaled to [0,1])
  - Channel conversion (RGB/RGBA/Grayscale)

## Error Handling

The application handles various error cases:
- Invalid file formats
- Corrupted images
- Model loading failures
- Image processing errors

## Customization

To modify the application:
1. Edit `app.py` to change the UI layout or add features
2. Update `requirements.txt` if you need different package versions
3. Replace `converted_keras/keras_model.h5` with your own trained model

## Notes

- The application uses TensorFlow's oneDNN custom operations, which may cause slight numerical differences due to floating-point round-off errors
- For optimal performance, use images with clear views of the fruits/vegetables
- The model works best with well-lit, centered images

## Troubleshooting

If you encounter issues:
1. Ensure all dependencies are correctly installed
2. Verify the model file exists in the correct location
3. Check that uploaded images are in supported formats
4. Ensure Python and pip are up to date



