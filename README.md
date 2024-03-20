# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

The task at hand involves developing a Convolutional Neural Network (CNN) that can accurately classify handwritten digits ranging from 0 to 9. This CNN should be capable of processing scanned images of handwritten digits, even those not included in the standard dataset.


The MNIST dataset is widely recognized as a foundational resource in both machine learning and computer vision. It consists of grayscale images measuring 28x28 pixels, each depicting a handwritten digit from 0 to 9. The dataset includes 60,000 training images and 10,000 test images, meticulously labeled for model evaluation. Grayscale representations of these images range from 0 to 255, with 0 representing black and 255 representing white. MNIST serves as a benchmark for assessing various machine learning models, particularly for digit recognition tasks. By utilizing MNIST, we aim to develop and evaluate a specialized CNN for digit classification while also testing its ability to generalize to real-world handwritten images not present in the dataset.

## Neural Network Model

![image](https://github.com/dharanielango/mnist-classification/assets/94530523/be6bb5ce-e137-48a0-a68c-42ac7046d6fb)



## DESIGN STEPS

#### STEP 1:
Preprocess the MNIST dataset by scaling the pixel values to the range [0, 1] and converting labels to one-hot encoded format.

#### STEP 2:
Build a convolutional neural network (CNN) model with specified architecture using TensorFlow Keras.

#### STEP 3:
Compile the model with categorical cross-entropy loss function and the Adam optimizer.

#### STEP 4:
Train the compiled model on the preprocessed training data for 5 epochs with a batch size of 64.

#### STEP 5:
Evaluate the trained model's performance on the test set by plotting training/validation metrics and generating a confusion matrix and classification report. Additionally, make predictions on sample images to demonstrate model inference.
## PROGRAM

##### Name:Dharani Elango
##### Register Number:212221230021

```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[2]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[2]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[400]
plt.imshow(single_image,cmap='gray')
y_train_onehot[200]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(keras.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters =32 , kernel_size =(3,3),activation="relu"))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dense(10,activation="softmax"))
model.summary()


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
model.fit(X_train_scaled ,y_train_onehot, epochs=10,batch_size=40, validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)
metrics.head()

print("212221230021 Dharani Elango")
metrics[['accuracy','val_accuracy']].plot()

print("212221230021 Dharani Elango")
metrics[['loss','val_loss']].plot()

print("212221230021 Dharani Elango")
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print("212221230021 Dharani Elango")
print(confusion_matrix(y_test,x_test_predictions))

print("212221230021 Dharani Elango")
print(classification_report(y_test,x_test_predictions))

img = image.load_img('/content/four.jpeg')
type(img)
print("212221230021 Dharani Elango")
plt.imshow(img)

img = image.load_img('/content/four.jpeg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

print("212221230021 Dharani Elango")
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print("212221230021 Dharani Elango")
print(x_single_prediction)

print("212221230021 Dharani Elango")
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

print("212221230021 Dharani Elango")
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
print("212221230021 Dharani Elango")
print(x_single_prediction)

```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/dharanielango/mnist-classification/assets/94530523/ed339d82-6c61-4474-91d3-16eec82291c8)

![image](https://github.com/dharanielango/mnist-classification/assets/94530523/82cdd385-5530-4a94-b41f-001c48820dfd)

![image](https://github.com/dharanielango/mnist-classification/assets/94530523/67bca5d2-b37d-4b5b-ad0e-37a997623541)



### Classification Report
![image](https://github.com/dharanielango/mnist-classification/assets/94530523/6f26d3ab-916f-4a61-a3bf-5a9ecea6099c)


### Confusion Matrix
![image](https://github.com/dharanielango/mnist-classification/assets/94530523/b4e7ed92-15b3-4d89-955b-cc8283401b0f)


### New Sample Data Prediction
![image](https://github.com/dharanielango/mnist-classification/assets/94530523/8fc67a40-b4f8-43f2-875f-a1adb21648c4)

![image](https://github.com/dharanielango/mnist-classification/assets/94530523/c36a33c5-f3af-456e-a152-178bef4f7b44)


![image](https://github.com/dharanielango/mnist-classification/assets/94530523/29ee21ee-b397-4777-9755-861e348a2cc1)


![image](https://github.com/dharanielango/mnist-classification/assets/94530523/f2b0280c-2d8c-481b-9a17-9088b1e89c46)

## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed successfully.
