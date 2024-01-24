# Satellite Image Analysis



https://github.com/XZIPX/Satellite-Image-Analysis/assets/96609166/1bee277d-9a41-4536-b9e9-51e605b1982b


## Overview
This project applies deep learning techniques to satellite image analysis, leveraging TensorFlow and PyTorch for classification and segmentation tasks. The application is built with Streamlit for ease of use and interactivity.

## Features
- **Classification**: Identifies various features in satellite images, such as 'Agricultural', 'Airplane', 'Buildings', etc.
- **Segmentation**: Performs precise building segmentation in satellite images.
- **Area Calculation**: Estimates the area of buildings in segmented images.

## Requirements
To run this project, you need the following libraries:
- TensorFlow
- PyTorch
- PIL (Python Imaging Library)
- NumPy
- Streamlit
- segmentation_models_pytorch
- torchvision

## Installation
Install all the required libraries using the following command:
```
pip install tensorflow torch PIL numpy streamlit segmentation_models_pytorch torchvision
```

## Usage
1. Start the application via Streamlit.
2. Upload a satellite image through the provided interface.
3. Input the scale (meters per pixel) for accurate area calculations.
4. The application will display both the classification results and the segmented image, along with the estimated area of buildings.

## Models
- Classification Model: Utilizes a MobileNet model adapted through transfer learning.
- Segmentation Model: Based on U-Net architecture with a ResNet50 backbone.

## Running the Application
To run the application, use the following command:
```
streamlit run main.py
```

## Contributing
Your contributions are always welcome! Feel free to submit issues or pull requests for improvements to the codebase.

## License
This project is licensed under the MIT License.
