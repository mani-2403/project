# STEP 1: Install TensorFlow (if needed, already available in Colab)
!pip install -q tensorflow

# STEP 2: Upload ZIP Dataset
from google.colab import files
import zipfile
import os

print(" Please upload your ZIP file (containing train/test folders)...")
uploaded = files.upload()

# STEP 3: Extract ZIP File
for zip_file in uploaded.keys():
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall("/content/")
        print(f" Extracted: {zip_file}")

# STEP 4: Set Paths Based on Extracted Folder Name
# Replace this with your actual extracted folder name if different
dataset_root = "/content/melanoma_cancer_dataset"
train_path = os.path.join(dataset_root, "train")
test_path = os.path.join(dataset_root, "test")

# STEP 5: CNN Training
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image preprocessing
img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_path, target_size=img_size, batch_size=batch_size, class_mode='categorical')

test_data = test_datagen.flow_from_directory(
    test_path, target_size=img_size, batch_size=batch_size, class_mode='categorical')

# CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(" Training started...")
model.fit(train_data, validation_data=test_data, epochs=5)
print(" Training complete!")

# STEP 6: Save Model
model_path = "/content/skin_cancer_model.h5"
model.save(model_path)
print(f" Model saved to: {model_path}")

# STEP 7: Download Model to Your Computer
files.download(model_path)
