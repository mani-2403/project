# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from google.colab import files
from PIL import Image
import matplotlib.pyplot as plt

# Load trained model from Drive
model_path = '/content/drive/MyDrive/skin_cancer_model.h5'  # Update path if needed
model = load_model(model_path, compile=False)

# Define class labels
class_names = ['benign', 'malignant']  # You can modify this if your model uses different classes

# Image upload prompt
print(" Please upload a skin lesion image (jpg/png) to test:")

# Upload image
uploaded = files.upload()

# Process and predict
for file_name in uploaded.keys():
    try:
        # Load and preprocess image
        img = Image.open(file_name).convert('RGB')
        img = img.resize((224, 224))  # Resize to model input shape
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using the model
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Show result
        plt.imshow(img)
        plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f" Error processing {file_name}: {str(e)}")
# Prevention tips dictionary
prevention_tips = {
    'benign': [
        " 1. Monitor the lesion regularly for changes in size, shape, or color.",
        " 2. Use sunscreen (SPF 30 or higher) daily.",
        " 3. Avoid excessive sun exposure between 10am and 4pm.",
        " 4. Wear protective clothing like hats and long sleeves.",
        " 5. Visit a dermatologist for routine check-ups."
    ],
    'malignant': [
        " 1. Seek immediate consultation with a dermatologist or oncologist.",
        " 2. Avoid direct sun exposure and always use high SPF sunscreen.",
        " 3. Do not try to self-treat or remove the lesion.",
        " 4. Follow proper medical guidance if a biopsy or surgery is recommended.",
        " 5. Maintain healthy skin habits and check for any new abnormal growths."
    ]
}

# Print prevention tips based on result
print("\n Suggested Prevention Tips:")
for tip in prevention_tips[predicted_class]:
    print(tip)

