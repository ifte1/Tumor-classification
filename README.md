# Brain Tumor Classification Using Deep Learning

This project uses convolutional neural networks and transfer learning to classify MRI brain scans into two categories: **Tumor** and **No Tumor**. It includes full data preparation, training using multiple deep learning models, and even real-world testing using a personal MRI scan.

---

## 🧠 Project Workflow

1. **Data Import**  
   Dataset of MRI images categorized into `yes/` (tumor) and `no/` (normal) folders.

2. **Exploratory Data Analysis (EDA)**  
   Class distribution, image samples, and imbalance analysis.

3. **Data Augmentation**  
   Applied transformations to increase dataset variability and balance classes.

4. **Data Preprocessing & Directory Setup**  
   Created train/val/test folders, renamed files, and loaded images using Keras generators.

5. **Train-Validation-Test Split**  
   80% training, 10% validation, 10% testing.

6. **Model Building (Transfer Learning)**

   ### ✅ Model 1: VGG19 (Frozen Layers)  
   Basic transfer learning — all convolutional layers frozen.

   ### ✅ Model 2: VGG19 (Last 2 Layers Unfrozen)  
   Slight fine-tuning on upper layers for better accuracy.

   ### ✅ Model 3: VGG19 (Fully Unfrozen)  
   Best performing model — fine-tuned using Model 2 weights.  
   **✔ Test Accuracy: 84%**  
            ### Got highest accuracy 
10/10 [==============================] - 4s 399ms/step - loss: 0.3825 - Valid accuracy: 0.8355
10/10 [==============================] - 4s 408ms/step - loss: 0.3696 - Test accuracy: 0.84

> Note: Results may vary depending on split and tuning.


- **Validation Accuracy**: ~83.6%
- **Training Loss** decreased steadily, indicating good convergence.

### ✅ Model 4: VGG19 + Adam Optimizer (Fully Unfrozen)  
Tried alternative optimizer — performance dropped compared to Model 3.

7. **Real-World Testing**  
Used a personal MRI scan (3 years old) of the author's father — tested successfully using the trained model.

8. **Hyperparameter Tuning**  
- Explored with Keras Tuner (Hyperband)
- Attempted ResNet50 and EfficientNetB0 architectures
- Found promising results but not outperforming Model 3 within current experiment scope

---

## 📊 Results

| Model                              | Test Accuracy         |
|------------------------------------|-----------------------|
| VGG19 (frozen)                     | 64% (10 epoch)        |
| VGG19 (partial unfreeze)           | 66% (10 epoch)        |
| **VGG19 (fully unfrozen)**         | **84% (10 epoch)** ✅ |
| VGG19 + Adam                       | Lower accuracy        |
| ResNet50- with 4 added dense layer | lower accuracy        |

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
# predicted correctly with 100% confidence level

## 📌 Future Work

- Add tumor segmentation to highlight affected area
- Convert notebook into a web app for deployment
- Extend to multi-class classification (tumor types)

## 👨‍💻 Author

Md Ifte Khairul Islam
MSc in Data Science
Based in Berlin | Focused on impactful machine learning and AI projects