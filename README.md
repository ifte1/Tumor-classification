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

