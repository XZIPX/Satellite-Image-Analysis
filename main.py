import tensorflow as tf
import torch
from PIL import Image
import numpy as np
import streamlit as st
import segmentation_models_pytorch as smp
from torchvision import transforms

classification_model = tf.keras.models.load_model("MobileNet_transfer_model.h5")
segmentation_model = torch.load("Unet_resnet50_model.pth.pth",
                                map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50', 'imagenet')

def predict_classification(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = classification_model.predict(image_array)[0]

    labels = ['Agricultural', 'Airplane', 'Baseball Diamond', 'Beach', 'Buildings', 'Chaparral',
              'Dense Residential', 'Forest', 'Freeway', 'Golf Course', 'Intersection', 'Medium Residential',
              'Mobile Home Park', 'Overpass', 'Parking Lot', 'River', 'Runway', 'Sparse Residential',
              'Storage Tanks', 'Tennis Court', 'Harbor']
    predicted_label = labels[np.argmax(predictions)]

    return {"label": predicted_label, "probability": max(predictions)}

def segment_image(image_path):
    original_image = Image.open(image_path).convert('RGB')
    original_size = original_image.size

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(original_image).unsqueeze(0)

    with torch.no_grad():
        pr_mask = segmentation_model(image_tensor)
    pr_mask = pr_mask.squeeze().cpu().numpy().round()

    pr_mask_resized = Image.fromarray(pr_mask.astype(np.uint8)).resize(original_size, resample=Image.BILINEAR)
    pr_mask_resized = np.array(pr_mask_resized)

    white_blue_mask_image = np.zeros_like(np.array(original_image))
    white_blue_mask_image[pr_mask_resized == 1] = [226, 48, 9]
    white_blue_mask_image[pr_mask_resized == 0] = [0, 0, 10]

    return white_blue_mask_image, pr_mask_resized

def calculate_area(segmented_mask, meters_per_pixel):
    building_pixels = np.sum(segmented_mask == 1)
    area_in_square_meters = building_pixels * (meters_per_pixel ** 2)
    return area_in_square_meters

st.title("Satellite Image Analysis")

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    meters_per_pixel = st.number_input("Enter the scale (meters per pixel):", min_value=0.0, value=0.1, step=0.01)
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    classification_prediction = predict_classification(uploaded_file)
    st.write("Label:", classification_prediction["label"])
    st.write("Probability:", classification_prediction["probability"])

    allowed_labels = ['Buildings', 'Dense Residential', 'Medium Residential', 'Sparse Residential', 'Mobile Home Park']
    if classification_prediction["label"] in allowed_labels:
        st.write("Performing Building Segmentation...")
        segmented_img, segmented_mask = segment_image(uploaded_file)
        st.image(segmented_img, caption='Segmented Image', use_column_width=True)

        area = calculate_area(segmented_mask, meters_per_pixel)
        st.write(f"Estimated area of buildings: {area:.2f} square meters")
