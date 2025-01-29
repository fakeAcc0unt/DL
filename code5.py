import tensorflow as tf 
from tensorflow.keras import layers, models 
from tensorflow.keras.datasets import cifar10 
from tensorflow.keras.utils import to_categorical 
# Step 1: Load and Preprocess the CIFAR-10 dataset 
(X_train, y_train), (X_test, y_test) = cifar10.load_data() 
# Normalize the images to the range [0, 1] 
X_train, X_test = X_train / 255.0, X_test / 255.0 
# One-hot encode the labels (10 classes in CIFAR-10) 
y_train = to_categorical(y_train, 10) 
y_test = to_categorical(y_test, 10) 
# Step 2: Build the CNN Model 
model = models.Sequential([ 
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),  # First convolutional 
layer 
layers.MaxPooling2D((2, 2)),  # Max pooling layer 
layers.Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer 
layers.MaxPooling2D((2, 2)),  # Max pooling layer 
layers.Conv2D(64, (3, 3), activation='relu'),  # Third convolutional layer 
layers.Flatten(),  # Flatten the output of the convolutional layers 
layers.Dense(64, activation='relu'),  # Fully connected layer 
layers.Dense(10, activation='softmax')  # Output layer (10 classes for CIFAR-10) 
]) 
# Step 3: Compile the model 
model.compile(optimizer='adam',  
loss='categorical_crossentropy',  
metrics=['accuracy']) 
# Step 4: Train the model 
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test)) 
# Step 5: Evaluate the model on the test set 
test_loss, test_accuracy = model.evaluate(X_test, y_test) 
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
