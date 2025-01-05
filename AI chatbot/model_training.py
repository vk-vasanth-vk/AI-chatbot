import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import json
import pickle

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents file
intents = json.load(open('intents.json'))

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Tokenize patterns and prepare data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# Save words and classes
if not os.path.exists('model'):
    os.makedirs('model')
pickle.dump(words, open('model/words.pkl', 'wb'))
pickle.dump(classes, open('model/classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    # Create bag of words
    bag = [1 if word in word_patterns else 0 for word in words]

    # Create output row
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

# Convert training data to NumPy arrays
training = np.array(training, dtype=object)

# Split input and output
train_x = np.array([t[0] for t in training], dtype=float)
train_y = np.array([t[1] for t in training], dtype=float)

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('model/chatbot_model.keras')
