from matplotlib.pyplot import imshow
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from Conv_NN import HappyModel
from tensorflow.keras.preprocessing import image
from kt_utils import load_dataset

# params
BATCH_SIZE = 32
EPOCH = 40

# load dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

happyModel = HappyModel(X_train.shape[1:])
happyModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
happyModel.summary()
happyModel.fit(x=X_train, y=Y_train, epochs=EPOCH, batch_size=BATCH_SIZE)

predictions = happyModel.evaluate(x=X_test, y=Y_test, batch_size=BATCH_SIZE, verbose=1)
print()
print("Loss = " + str(predictions[0]))
print("Test Accuracy = " + str(predictions[1]))


### START CODE HERE ###
img_path = 'current_image/current_image.jpg'
### END CODE HERE ###
img = image.load_img(img_path, target_size=(64, 64))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(happyModel.predict(x))