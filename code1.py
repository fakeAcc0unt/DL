import tensorflow as tf 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Flatten 
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.utils import to_categorical 
 
# Load and preprocess the MNIST dataset 
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
 
# Normalize the data to the range [0, 1] 
x_train = x_train.astype('float32') / 255.0 
x_test = x_test.astype('float32') / 255.0 
 
# Convert labels to one-hot encoding 
y_train = to_categorical(y_train, 10) 
y_test = to_categorical(y_test, 10) 
 
# Build the feedforward neural network model 
model = Sequential([ 
    Flatten(input_shape=(28, 28)),  # Flatten 28x28 images into a vector 
    Dense(128, activation='relu'),  # Hidden layer with 128 neurons 
    Dense(64, activation='relu'),   # Hidden layer with 64 neurons 
    Dense(10, activation='softmax') # Output layer for 10 classes 
]) 
 
# Compile the model 
model.compile( 
    optimizer='adam',  # Optimizer 
    loss='categorical_crossentropy',  # Loss function for multi-class classification 
    metrics=['accuracy']  # Metrics to monitor during training 
) 
 
# Train the model 
model.fit( 
    x_train, y_train,  # Training data and labels 
    epochs=10,  # Number of epochs 
    batch_size=32,  # Batch size 
    validation_data=(x_test, y_test)  # Validation data 
) 
 
# Evaluate the model on test data 
loss, accuracy = model.evaluate(x_test, y_test) 
print(f"Test Loss: {loss:.4f}") 
print(f"Test Accuracy: {accuracy:.4f}")
