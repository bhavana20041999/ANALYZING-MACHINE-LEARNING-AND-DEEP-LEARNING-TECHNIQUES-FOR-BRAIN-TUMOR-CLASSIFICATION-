import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import joblib
import os
import json

# Title
st.title("Brain Tumor Classification System")
st.header("Upload an Image for Prediction (MRI, CT, or PET)")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Feature extractor (ResNet18 frozen)
@st.cache_resource
def get_feature_extractor():
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = torch.nn.Identity()  # Remove classifier
    model.eval()
    return model

extractor = get_feature_extractor()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load saved SVM model and scaler
@st.cache_resource
def load_model_and_scaler(modality, task):
    model_path = f"models/svm_resnet_{modality}_{task}.pkl"
    scaler_path = f"models/scaler_resnet_{modality}_{task}.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        svm = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return svm, scaler
    else:
        st.error(f"Model files not found for {modality} {task}")
        return None, None

# UI inputs
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)
with col1:
    modality = st.selectbox("Select Modality", ["mri", "ct", "pet"])
with col2:
    task = st.selectbox("Select Task", ["binary", "multiclass"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # SMALL FIXED IMAGE SIZE – clean and professional
    st.image(image, caption="Uploaded Image", width=300)  # 300px – small & consistent

    if st.button("Classify Tumor"):
        with st.spinner("Processing image and predicting..."):
            # Extract features
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                features = extractor(img_tensor).cpu().numpy().flatten()
            
            # Load model and scaler
            svm, scaler = load_model_and_scaler(modality, task)
            if svm is None:
                st.stop()
            
            # Scale and predict
            features_scaled = scaler.transform(features.reshape(1, -1))
            prediction = svm.predict(features_scaled)[0]
            probabilities = svm.predict_proba(features_scaled)[0]
            confidence = probabilities[prediction] * 100

            # Get label
            if task == "binary":
                label = class_names["binary"][prediction]
            else:
                label = class_names["multiclass"][prediction]

            # Display result
            st.success(f"**Prediction: {label}**")
            st.info(f"Confidence: {confidence:.2f}%")

            # Show probability bar
            prob_df = pd.DataFrame({
                "Class": class_names[task],
                "Probability (%)": probabilities * 100
            })
            st.bar_chart(prob_df.set_index("Class"))

# Sidebar with project info
st.sidebar.header("Project Summary")
st.sidebar.write("**Best Model:** ResNet18 Features + SVM")
st.sidebar.write("**Performance Highlights:**")
st.sidebar.write("• MRI Binary: 99.77%")
st.sidebar.write("• MRI Multiclass: 97.51%")
st.sidebar.write("• CT Binary: 96.97%")
st.sidebar.write("• PET: 100.00% (both tasks)")
st.sidebar.write("Developed using transfer learning on public datasets.")
st.sidebar.caption("© Copyright 2026 PARISE BHAVANA | Sheffield Hallam University")
st.caption("© Copyright 2026 PARISE BHAVANA  | MSc Big Data Analytics | Sheffield Hallam University")

st.sidebar.info("Upload any brain scan image and select modality/task to get instant classification!")