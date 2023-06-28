import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Path to the pre-trained sentiment analysis model
model_path = "saved_model"

# Load the pre-trained segmentation model
segmentation_model = tf.keras.models.load_model(model_path)

# Target image shape
TARGET_SHAPE = (256, 256)

# Define image segmentation function
def segment_image(img:np.ndarray):
    # Original image shape
    ORIGINAL_SHAPE = img.shape

    # Check if the image is RGB and convert if not
    if len(ORIGINAL_SHAPE) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Resize the image to TARGET_SHAPE
    img = cv2.resize(img, TARGET_SHAPE)
    
    # Add a batch dimension
    img = np.expand_dims(img, axis=0)
    
    # Predict the segmentation mask
    mask = segmentation_model.predict(img)

    # Remove the batch dimension
    mask = np.squeeze(mask, axis=0)

    # Convert to labels 
    mask = np.argmax(mask, axis=-1)
    
    # Convert to uint8
    mask = mask.astype(np.uint8)

    # Resize to original image shape
    mask = cv2.resize(mask, (ORIGINAL_SHAPE[1], ORIGINAL_SHAPE[0]))

    return mask

def overlay_mask(img, mask, alpha=0.5):
    # Define color mapping
    colors = {
        0: [255, 0, 0],   # Class 0 - Red
        1: [0, 255, 0],   # Class 1 - Green
        2: [0, 0, 255]    # Class 2 - Blue
        # Add more colors for additional classes if needed
    }

    # Create a blank colored overlay image
    overlay = np.zeros_like(img)

    # Map each mask value to the corresponding color
    for class_id, color in colors.items():
        overlay[mask == class_id] = color

    # Blend the overlay with the original image
    output = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

    return output


# The main function
def transform(img):
    mask=segment_image(img)
    blended_img = overlay_mask(img, mask)
    return blended_img


# Create the Gradio app
app = gr.Interface(
    fn=transform, 
    inputs=gr.Image(label="Input Image"), 
    outputs=gr.Image(label="Image with Segmentation Overlay"), 
    title="Image Segmentation on Pet Images",
    description="Segment image of a pet animal into three classes: background, pet, and boundary.",
    examples=[
        "example_images/img1.jpg",
        "example_images/img2.jpg",
        "example_images/img3.jpg"
    ]
)
                   
# Run the app
app.launch()