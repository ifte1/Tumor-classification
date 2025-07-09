# Tumor-classification

## Dataset = keggle

## Step followed: 

### 1. Data import
### 2. EDA(Exploratory Data Analysis)
### 3. Data Augmentation
### 4. Data Preprocessing
### 5. Image Loading
### 6. Plotting the images
### 7. Making directories to store data
### 8. Train Test and validation split
### 9. Model 1: using vgg19 (freezing all the previous parameters)
  ###  Model 2: unfreezing last two layer and using previous model parameters
  ###  Model 3: Unfreezing the entire network and using model 2 weights
            ### Got highest accuracy 
10/10 [==============================] - 4s 399ms/step - loss: 0.3825 - Valid accuracy: 0.8355
10/10 [==============================] - 4s 408ms/step - loss: 0.3696 - Test accuracy: 0.84
  ###  Test : Real MRI image of my father that i captured almost 3 years ago
  ###  Model 4: Model with Adam optimizer and Unfreezing the entire network
### 10. Hyper parameter tuning



# Brain Tumor Classification Using Deep Learning

This project implements a deep learning-based approach for classifying brain MRI images into two categories: **Tumor** and **No Tumor**. The project explores multiple transfer learning models including VGG19, ResNet50, and EfficientNetB0. It also includes hyperparameter tuning, data augmentation, and a real-world test case using an actual MRI scan.

---

## 🔍 Project Overview

- **Objective**: Accurately classify MRI images as containing a brain tumor or not.
- **Techniques Used**:
  - Transfer Learning (VGG19, ResNet50, EfficientNetB0)
  - Data Preprocessing and Augmentation
  - Hyperparameter Tuning (Keras Tuner)
  - Custom directory setup and renaming scripts
  - Inference on real-life MRI scan

---

## 📁 Dataset

The dataset is divided into two folders:

- `yes/` - MRI scans with tumors
- `no/` - MRI scans without tumors

Each image was renamed for consistency. The data was later split into **80% training**, **10% validation**, and **10% testing**.

---

## 🧠 Models Trained

### Model 1
- VGG19 with all layers frozen
- Baseline transfer learning performance

### Model 2
- VGG19 with last two convolutional blocks unfrozen
- Improved generalization

### Model 3
- VGG19 fully unfrozen
- Fine-tuned on Model 2 weights

### Model 4
- VGG19 + Adam optimizer with full unfreezing
- Underperformed compared to others

### Model 5
- ResNet50 with Keras Tuner for hyperparameter tuning
- Best performing model

### Model 6
- EfficientNetB0 with Keras Tuner
- Slightly lower performance than ResNet50

---

## 📊 Results
# Model 3: Unfreezing the entire network and using model 2 weights
            ### Got highest accuracy 
10/10 [==============================] - 4s 399ms/step - loss: 0.3825 - Valid accuracy: 0.8355
10/10 [==============================] - 4s 408ms/step - loss: 0.3696 - Test accuracy: 0.84

> Note: Results may vary depending on split and tuning.

---

## 🔧 Technologies Used

- Python
- TensorFlow & Keras
- OpenCV
- Matplotlib & Seaborn
- Keras Tuner
- Scikit-learn


---

## 🧪 Real-World Test Case

The model was tested on a personal MRI scan (3 years old) of my father, demonstrating its practical application.

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('best_model.h5')
img = image.load_img('path_to_mri_image.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

prediction = model.predict(x)
print("Prediction:", "Tumor" if prediction[0][0] > 0.5 else "No Tumor")
```

## 📌 Future Work

- Add tumor segmentation to highlight affected area
- Convert notebook into a web app for deployment
- Extend to multi-class classification (tumor types)

## 👨‍💻 Author

Md Ifte Khairul Islam
MSc in Data Science
Based in Berlin | Focused on impactful machine learning and AI projects