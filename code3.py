import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

max_words = 10000 # Consider only the top 10,000 most frequent words
max_len = 200 # Limit review length to 200 words

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_words)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

model = Sequential([
    Embedding(input_dim=max_words, output_dim=128), # Embedding layer for word representation
    GlobalAveragePooling1D(), # Global average pooling layer to reduce dimensions
    Dense(64, activation='relu'), # Fully connected layer
    Dense(1, activation='sigmoid') # Output layer (binary classification)
])

model.compile(optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy*100:.2f}")
