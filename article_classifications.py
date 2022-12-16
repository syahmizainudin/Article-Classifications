# %%
import os
import re
import json
import nltk
import pickle
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, plot_model
from keras import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Download nltk data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# %% 1. Data loading
CSV_URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'

df = pd.read_csv(CSV_URL)

# %% 2. Data inspection
df.head(10)
df.tail(10)

df.info()
df.describe()

df.isna().sum() # no NaN values
df.duplicated().sum() # 99 duplicates need to be remove

df['text'][0] # text in the data had mostly been cleaned, but the unnecessary symbols can be removed for better efficiency

# Visualize the distribution of category
plt.figure()
sns.displot(df['category'])
plt.show()

# %% 3. Data cleaning
# Remove duplicates
df = df.drop_duplicates()
df.duplicated().sum()

# Remove unnecessary information
STOPWORDS = set(stopwords.words('english'))

def clean_text(text, stop_words=STOPWORDS):
    lemmatizer = WordNetLemmatizer()
    filter = '[^\w]'
    text = [lemmatizer.lemmatize(w) for w in word_tokenize(text.lower()) if w not in stop_words]
    text = ' '.join(text)
    return re.sub(filter, ' ', text)

df['text'] = df['text'].apply(clean_text)
df['text'][0]

# Summary for data in text column
np.sum(df['text'].str.split().str.len())
np.mean(df['text'].str.split().str.len())
np.median(df['text'].str.split().str.len())
np.max(df['text'].str.split().str.len())

# %% 4. Features selection
# Define the features and targets
features = df['text']
targets = df['category']

# %% 5. Data pre-processing
# Tokenization
NUM_WORDS = 5000
OOV_TOKEN = '<OOV>'

tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(features)
feature_sequences = tokenizer.texts_to_sequences(features)
tokenizer.word_index

# Padding & truncating
feature_sequences = pad_sequences(feature_sequences, maxlen=250, padding='post', truncating='post')
feature_sequences[785]

# Expand the dimension of the features and targets
feature_sequences = np.expand_dims(feature_sequences, -1)
targets = np.expand_dims(targets, -1)

# Encode targets
ohe = OneHotEncoder(sparse=False)
targets = ohe.fit_transform(targets)

# Train-test split
SEED = 12345

X_train, X_test, y_train, y_test = train_test_split(feature_sequences, targets, random_state=SEED)

# %% Model development
# Define Sequential model
EMBEDDING_DIM = 64

model = Sequential()
model.add(Embedding(NUM_WORDS, EMBEDDING_DIM))
model.add(Bidirectional(LSTM(EMBEDDING_DIM)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Model summary
model.summary()
plot_model(model, to_file=os.path.join(os.getcwd(), 'resources', 'model.png'), show_shapes=True, show_layer_names=True)

# Model compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='acc')

# Define callbacks
LOG_DIR = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tb = TensorBoard(log_dir=LOG_DIR)
es = EarlyStopping(monitor='val_acc', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_lr=0.001)
mc = ModelCheckpoint(filepath=os.path.join(os.getcwd(), 'temp', 'checkpoint'), save_weights_only=True, 
                     monitor='val_acc', mode='max', save_best_only=True)

# Model training
EPOCHS = 10
BATCH_SIZE = 64

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, 
                    batch_size=BATCH_SIZE, callbacks=[tb, es, reduce_lr, mc])

# %% Model evaluation
# Prediction with the model
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print('Classification report:\n', classification_report(y_true, y_pred))

# %% Model saving
SAVE_PATH = os.path.join(os.getcwd(), 'saved_models')

# Save tokenizer
with open(os.path.join(SAVE_PATH, 'tokenizer.json'), 'w') as f:
    json.dump(tokenizer.to_json(), f)

# Save encoder
with open(os.path.join(SAVE_PATH, 'OneHotEncoder.pkl'), 'wb') as f:
    pickle.dump(ohe, f)

# Save model
model.save(filepath=os.path.join(SAVE_PATH, 'model.h5'))
# %%
