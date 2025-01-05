import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re, string, time
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, AveragePooling1D, Dense, Flatten, Dropout
from keras.optimizers import Adam

import nltk
nltk.download('punkt')
nltk.download('stopwords')

def dataset():
    data = pd.read_excel("Dataset.xlsx")
    jumlah = data['Label'].value_counts()
    return data, jumlah

def hitung_label(data):
    data['Label'] = data['Label'].str.lower()
    jumlah = data['Label'].value_counts()
    return jumlah

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()

def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'RT[\s]', '', text)
    text = re.sub(r"http\S+", '', text)
    text = re.sub(r'[0-9]+', '', text)

    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip(' ')
    return text

def casefoldingText(text):
    text = text.lower()
    return text

def tokenizingText(text):
    text = word_tokenize(text)
    return text

def filteringText(text):
    stopwords_indonesian = stopwords.words('indonesian')
    stopwords_indonesian.remove('ada')
    stopwords_indonesian.remove('kurang')
    stopwords_indonesian.remove('tidak')
    filtered = []
    for txt in text:
        if txt not in stopwords_indonesian:
            filtered.append(txt)
    text = filtered
    return text

def stemmingText(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = [stemmer.stem(word) for word in text]
    return text

def join_text_list(texts):
    return ' '.join([text for text in texts])

def preprocess(text):
    text = cleaningText(text)
    text = casefoldingText(text)
    text = tokenizingText(text)
    text = filteringText(text)
    text = stemmingText(text)
    return text

def tf_idf(data):
    vectorizer = TfidfVectorizer()
    tfidf_df = pd.DataFrame(vectorizer.fit_transform(data['Text']).toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_vectors = vectorizer.fit_transform(data['Text'])
    return tfidf_df, tfidf_vectors, vectorizer.vocabulary_

def models(tf_idf, ac_1, ac_2, vocab):
    word_index = vocab
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, 17, input_length=tf_idf.shape[1]))
    model.add(Conv1D(128, 8, activation=ac_1))
    model.add(MaxPooling1D(8))
    model.add(Dropout(0.3))
    model.add(Conv1D(64, 4, activation=ac_2))
    model.add(AveragePooling1D(4))
    model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(3, activation="softmax"))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model

def format_waktu(seconds):
    menit, detik = divmod(seconds, 60)
    return f"{int(menit)} menit {int(detik)} detik"

def fold(tf_idf, ac_1, ac_2, vocab, k, tf_idf_vector, data, epoch, batch):
    model = models(tf_idf, ac_1, ac_2, vocab)
    start_time = time.time()
    kf = KFold(n_splits=k, shuffle=True)
    encoder = LabelEncoder()
    hasil_list = []
    
    for fold, (train_index, test_index) in enumerate(kf.split(tf_idf_vector)):
        X_train, X_test = tf_idf_vector[train_index].toarray(), tf_idf_vector[test_index].toarray()
        y_train, y_test = data['Label'][train_index], data['Label'][test_index]
        encoder.fit(y_train)
        encoded_Y_train = encoder.transform(y_train)
        encoded_Y_test = encoder.transform(y_test)
        y_train = to_categorical(encoded_Y_train, num_classes=3)
        y_test = to_categorical(encoded_Y_test, num_classes=3)
        history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch, verbose=1)
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        loss = model.evaluate(X_test, y_test, verbose=0)[0]
        accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred) * 100
        precision = precision_score(np.argmax(y_test, axis=1), y_pred, average='macro') * 100
        recall = recall_score(np.argmax(y_test, axis=1), y_pred, average='macro') * 100
        f1 = f1_score(np.argmax(y_test, axis=1), y_pred, average='macro') * 100
        
        hasil_list.append({
            "Fold": f"{fold + 1}",
            "Loss": f"{round(loss * 100, 2)}%",
            "Akurasi": f"{accuracy:.2f}%",
            "Presisi": f"{precision:.2f}%",
            "Recall": f"{recall:.2f}%",
            "F1 Score": f"{f1:.2f}%",
        })

    hasil = pd.DataFrame(hasil_list)
    r_Loss = hasil["Loss"].replace('%', '', regex=True).astype('float').mean()
    r_Akurasi = hasil["Akurasi"].replace('%', '', regex=True).astype('float').mean()
    r_Presisi = hasil["Presisi"].replace('%', '', regex=True).astype('float').mean()
    r_Recall = hasil["Recall"].replace('%', '', regex=True).astype('float').mean()
    r_F1 = hasil["F1 Score"].replace('%', '', regex=True).astype('float').mean()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    formatted_time = format_waktu(elapsed_time)

    return hasil, r_Loss, r_Akurasi, r_Presisi, r_Recall, r_F1, formatted_time
