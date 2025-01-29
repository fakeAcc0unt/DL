import tensorflow as tf 
from tensorflow.keras import layers, models 
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.utils import to_categorical 
# Step 1: Load and Preprocess the MNIST dataset 
(X_train, y_train), (X_test, y_test) = mnist.load_data() 
# Normalize the images to [0, 1] by dividing by 255 
X_train = X_train / 255.0 
X_test = X_test / 255.0 
# Reshape the data to (batch_size, height, width, channels) for CNN input 
X_train = X_train.reshape(-1, 28, 28, 1) 
X_test = X_test.reshape(-1, 28, 28, 1) 
# One-hot encode the labels (digits 0-9) 
y_train = to_categorical(y_train, 10) 
y_test = to_categorical(y_test, 10) 
# Step 2: Build the CNN Model 
model = models.Sequential([ 
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), 
layers.MaxPooling2D((2, 2)), 
layers.Conv2D(64, (3, 3), activation='relu'), 
layers.MaxPooling2D((2, 2)), 
layers.Conv2D(64, (3, 3), activation='relu'), 
layers.Flatten(), 
layers.Dense(64, activation='relu'), 
layers.Dense(10, activation='softmax')  # Output layer with 10 classes (0-9) 
]) 
# Step 3: Compile the model 
model.compile(optimizer='adam',  
loss='categorical_crossentropy',  
metrics=['accuracy']) 
# Step 4: Train the model 
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test)) 
# Step 5: Evaluate the model on the test set 
test_loss, test_accuracy = model.evaluate(X_test, y_test) 
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
