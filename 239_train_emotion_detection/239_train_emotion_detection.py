
"""
Train a deep learning model on facial emotion detection

Dataset from: https://www.kaggle.com/msambare/fer2013
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
import os
from matplotlib import pyplot as plt
import numpy as np

IMG_HEIGHT=48 
IMG_WIDTH = 48
batch_size=32

trainDataDir='/Users/macuser/Downloads/cvdata/train/'
validationDataDir='/Users/macuser/Downloads/cvdata/test/'

trainDatagen = ImageDataGenerator(
					rescale=1./255,
					rotation_range=30,
					shear_range=0.3,
					zoom_range=0.3,
					horizontal_flip=True,
					fill_mode='nearest')

validationDatagen = ImageDataGenerator(rescale=1./255)

trainGenerator = trainDatagen.flow_from_directory(
					trainDataDir,
					color_mode='grayscale',
					target_size=(IMG_HEIGHT, IMG_WIDTH),
					batch_size=batch_size,
					class_mode='categorical',
					shuffle=True)

validationGenerator = validationDatagen.flow_from_directory(
							validationDataDir,
							color_mode='grayscale',
							target_size=(IMG_HEIGHT, IMG_WIDTH),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)

#Verify our generator by plotting a few faces and printing corresponding labels
classLabels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']

img, label = trainGenerator.__next__()

import random

i=random.randint(0, (img.shape[0])-1)
image = img[i]
labl = classLabels[label[i].argmax()]
plt.imshow(image[:,:,0], cmap='gray')
plt.title(labl)
plt.show()
##########################################################


###########################################################
# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))

model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())


trainPath = "/Users/macuser/Downloads/cvdata/train/"
testPath = "/Users/macuser/Downloads/cvdata/test"

numTrainImgs = 0
for root, dirs, files in os.walk(trainPath):
    numTrainImgs += len(files)
    
numTestImgs = 0
for root, dirs, files in os.walk(testPath):
    numTestImgs += len(files)


epochs=50

history=model.fit(trainGenerator,
                steps_per_epoch=numTrainImgs//batch_size,
                epochs=epochs,
                validation_data=validationGenerator,
                validation_steps=numTestImgs//batch_size)

model.save("/Users/macuser/Downloads/cvdata/FER_model.h5")

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
#acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

####################################################################
from keras.models import load_model


#Test the model
my_model = load_model('/Users/macuser/Downloads/cvdata/FER_model.h5', compile=False)

#Generate a batch of images
test_img, test_lbl = validationGenerator.__next__()
predictions=my_model.predict(test_img)

predictions = np.argmax(predictions, axis=1)
test_labels = np.argmax(test_lbl, axis=1)

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, predictions))

#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, predictions)
#print(cm)
import seaborn as sns
sns.heatmap(cm, annot=True)

classLabels=['Surprise','Angry','Disgust', 'Fear', 'Happy','Neutral','Sad']
#Check results on a few select images
n=random.randint(0, test_img.shape[0] - 1)
image = test_img[n]
orig_labl = classLabels[test_labels[n]]
pred_labl = classLabels[predictions[n]]
plt.imshow(image[:,:,0], cmap='gray')
plt.title("Original label is:"+orig_labl+" Predicted  label is: "+ pred_labl)
plt.show()
