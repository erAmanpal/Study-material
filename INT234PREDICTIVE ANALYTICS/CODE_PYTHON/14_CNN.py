import numpy as np
# Load the MNIST data from local file
data = np.load('mnist.npz')
lst = data.files
print(lst)  #"['x_test', 'x_train', 'y_train', 'y_test']"

# data is a NumPy .npz file — it's not a tuple or list of tuples.
#It’s a dictionary-like object with named arrays.
#Not valid ---->>> (x_train, y_train), (x_test, y_test) = data

# Extract training and test sets
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

# Reduce the size as to demonstrate the concept
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:1000]
y_test =y_test[:1000]


# Confirm shapes
##print("Train images:", x_train.shape)   #(60000, 28, 28)
##print("Train labels:", y_train.shape)   #(60000,)
##print("Test images:", x_test.shape)     #(10000, 28, 28)
##print("Test labels:", y_test.shape)     #(10000,)
# 1. Reshaping prepares the data for CNN input format.
# 2. Normalization improves training speed and stability.
# 3. One-hot encoding matches the output format of the final softmax layer in your CNN.

# Reshape and normalize training IMAGES
# CNNs expect input as (samples, height, width, channels)
# samples -1: placeholder (Figure this dimension out automatically)
# height and width: size of each image (28×28 pixels)
# channels: number of color channels, 1->grayscale 3->RGB
# / 255.0: px 0=black 255=white; Normalize pixel values to [0, 1] for better training
#Most optimizers (like Adam) perform better when inputs are in a small,
#consistent range, typically between 0 and 1
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0



from tensorflow.keras.utils import to_categorical
# Converts integer LABELS (e.g., 5) into one-hot vectors:
# np.uint8(5) to array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])
#This is needed for multi-class classification with softmax output
# y_train_cat.shape: (60000, 10) => 10 columns for digits 0–9
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

##Conv2D: Finds patterns in small image patches (like edges or curves).
##MaxPooling2D: Shrinks the image to reduce complexity.
##Flatten: Turns a 2D image into a 1D list.
##Dense: A fully connected layer that makes decisions.

# You’re starting to build a stack of layers

## Conv2D(32, (3,3)): Adds 32 filters (small 3×3 windows) that slide over the image to detect features like edges or corners.
## activation='relu': Applies a function that keeps only positive values (helps the network learn better).
## input_shape=(28,28,1): Each input image is 28×28 pixels, with 1 color channel (grayscale).
## MaxPooling2D(2,2): Looks at 2×2 blocks in the image and keeps only the strongest signal (the maximum value).
## Flatten(): Takes the 2D image data and flattens it into a 1D list.
## Dense(64, activation='relu'): A fully connected layer with 64 neurons.
#  Each neuron learns to recognize combinations of features (like loops, lines, etc.).
#  Uses ReLU to keep only positive signals.
## Dense(10, activation='softmax'): 
#   Final layer with 10 neurons — one for each digit (0 to 9).
#   Softmax turns the outputs into probabilities that add up to 1.

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Adam optimizer adjusts weights efficiently.
# Categorical crossentropy is used for multi-class classification.
# Trains for 2 epochs (passes through the data 3 times).
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train_cat, epochs=2, validation_split=0.1)

# Tests the model on unseen data.
# Prints accuracy (usually >98% for MNIST!)
loss, acc = model.evaluate(x_test, y_test_cat)
print("CNN Accuracy:", acc)






