#!/usr/bin/env python
# coding: utf-8

# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns


# Sadece 1 ile etiketlenen (ırkçılık içeren) tweetler ve ilgili sütunlar seçiliyor
racist_tweets_analysis = data[data['label'] == 1][['date', 'user_created', 'user_followersCount']]

# Tarih sütunlarını datetime formatına dönüştürme
racist_tweets_analysis['date'] = pd.to_datetime(racist_tweets_analysis['date'])
racist_tweets_analysis['user_created'] = pd.to_datetime(racist_tweets_analysis['user_created'], errors='coerce')

# Tarih sütunlarını aynı zaman dilimine getirme
racist_tweets_analysis['date'] = racist_tweets_analysis['date'].dt.tz_localize(None)
racist_tweets_analysis['user_created'] = racist_tweets_analysis['user_created'].dt.tz_localize(None)

# Hesabın kaç gün önce açıldığını hesaplama
racist_tweets_analysis['account_age_days'] = (racist_tweets_analysis['date'] - racist_tweets_analysis['user_created']).dt.days

# Hesap yaşı ve takipçi sayısına göre verilerin dağılımını inceleme
plt.figure(figsize=(12, 6))

# Hesap yaşı ve takipçi sayısı arasındaki ilişkiyi gösteren bir scatter plot
sns.scatterplot(x='account_age_days', y='user_followersCount', data=racist_tweets_analysis)

plt.xlabel('Account Age (Days)')
plt.ylabel('Number of Followers')
plt.title('Relationship Between Account Age and Number of Followers for Tweets Containing Racism')
plt.yscale('log')  # We use the logarithmic scale for the number of followers
plt.grid(True)
plt.show()


# In[18]:


from datetime import datetime

# 'user_created' sütunundaki tarihleri datetime formatına dönüştürme
data['user_created_datetime'] = pd.to_datetime(data['user_created'])

# Dönüştürülmüş tarihlerin ilk birkaçını gösterme
data[['user_created', 'user_created_datetime']].head()


# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns

# Irkçı ve ırkçı olmayan tweetlerin kullanıcı hesap oluşturma tarihlerini ayrı ayrı al
racist_tweets = data[data['label'] == 1]['user_created_datetime']
non_racist_tweets = data[data['label'] == 0]['user_created_datetime']

# Grafik oluşturma
# Creating the graph
plt.figure(figsize=(12, 6))
sns.histplot(racist_tweets, color="red", label="Racist Tweets", kde=True)
sns.histplot(non_racist_tweets, color="blue", label="Non-Racist Tweets", kde=True)
plt.title("Distribution of Account Creation Dates (Racist vs Non-Racist Tweets)")
plt.xlabel("Account Creation Date")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# In[2]:


# Gerekli kütüphaneleri yükleme
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense
import networkx as nx
import torch
from torch_geometric.nn import GCNConv
from torch.nn import LSTM, Linear, Module
from IPython.display import HTML


# Veri kümesini yükleme
file_path = '/Users/sulekaya/Downloads/2023/TEZ/kod/son_veri.xlsx'
data = pd.read_excel(file_path)

# Stopwords yükleme
with open('/Users/sulekaya/Downloads/2023/TEZ/kod/turkce-stop-words.txt', 'r', encoding='utf-8') as file:
    turkish_stopwords = file.read().splitlines()

import re
import pandas as pd

def clean_text(text):
    # NaN değerler için kontrol
    if pd.isna(text):
        return ""

    # URL'leri, kullanıcı adlarını ve özel karakterleri temizleme
    text = re.sub(r'http\S+', '', text)  # URL'leri temizle
    text = re.sub(r'@\w+', '', text)     # Kullanıcı adlarını temizle
    text = re.sub(r'\W', ' ', text)      # Özel karakterleri temizle

    # Emojileri temizleme
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # Metni küçük harflere çevirme
    text = text.lower()

    # Durak kelimeleri kaldırma
    text = ' '.join([word for word in text.split() if word not in turkish_stopwords])

    # Fazla boşlukları kaldırma
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Metinleri temizleme ve durak kelimeleri kaldırma
data['cleaned_content'] = data['content'].apply(clean_text)



# Metinleri temizleme
data['cleaned_content'] = data['content'].apply(clean_text)


# In[3]:


# 'label' sütunundaki 0 ve 1 değerlerinden eşit sayıda örnek seçme
sample_0 = data[data['label'] == 0].sample(4)
sample_1 = data[data['label'] == 1].sample(4)

# Seçilen örnekleri birleştirme
sampled_data = pd.concat([sample_0, sample_1])

# 'content' ve 'label' sütunlarını içeren DataFrame'i oluşturma
content_label_table = sampled_data[['content', 'label']]

# HTML tablosu oluşturma
html_table = content_label_table.to_html(index=False, escape=False, border=1)

# HTML tablosunu gösterme
HTML(html_table)


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns 
# Metin uzunluklarını hesaplama
data['content_length'] = data['content'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)

# Metin uzunluklarının dağılımını görselleştirme
plt.figure(figsize=(10, 6))
sns.histplot(data['content_length'], bins=50, kde=True)
plt.title('Distribution of Text Lengths')
plt.xlabel('Text Length (Word Count)')
plt.ylabel('Frequency')
plt.show()


# In[5]:


from wordcloud import WordCloud

# Tüm metinleri birleştirme
# NaN değerleri boş string ile değiştirerek ve her bir değeri stringe çevirerek
all_text = ' '.join([str(text) for text in data['cleaned_content'] if pd.notnull(text)])

# Kelime bulutu oluşturma
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Kelime bulutunu görselleştirme
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[6]:


import matplotlib.pyplot as plt
from collections import Counter

# Türkçe kelimeler ve İngilizce karşılıkları
kelime_cevirileri = {
    'türk': 'Turkish',
    'türkiye': 'Turkey',
    'abd': 'USA',
    'suriye': 'Syria',
    'ermeniler': 'Armenians',
    'arap': 'Arabian',
    'ermeni': 'Armenian',
    'kürt': 'Kurdish',
    'ermenistan': 'Armenia',
    'yunan': 'Greek'
}

# Metinlerdeki tüm kelimeleri bir listeye ekleyin
all_words = ' '.join(data['cleaned_content']).split()

# Kelime frekanslarını hesaplayın
word_freq = Counter(all_words)

# En sık kullanılan 10 kelimeyi seçin
most_common_words = word_freq.most_common(10)

# Kelimeler ve frekansları ayrı listelere ayırın
words, freqs = zip(*most_common_words)

# Pasta grafiğini oluşturun
plt.figure(figsize=(10, 8))
patches, texts, autotexts = plt.pie(freqs, labels=words, startangle=140, colors=plt.cm.Paired.colors,
                                    autopct=lambda pct: "{:.1f}%\n({:d})".format(pct, int(pct/100.*sum(freqs))))

# Autotexts'in font boyutunu ayarlayın
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(10)

plt.title('The 10 Most Frequently Used Words and Their Frequencies')
plt.axis('equal')  # This ensures the pie chart is circular

# Açıklama kutusunu ekle ve konumunu ayarla
translated_labels = [f'{w} ({kelime_cevirileri.get(w, "N/A")}): {f}' for w, f in zip(words, freqs)]
plt.legend(patches, translated_labels, title="Word Counts", loc="center left", bbox_to_anchor=(1, 0.5))

# Görseli göster
plt.show()


# In[7]:


import matplotlib.pyplot as plt

# Etiketlerin frekanslarını hesaplama
label_counts = data['label'].value_counts()

# Pasta grafiğini oluşturma
plt.figure(figsize=(8, 6))
plt.pie(label_counts, labels=label_counts.index, autopct=lambda p: '{:.1f}%\n({:d})'.format(p, int(p/100*label_counts.sum())), startangle=140, colors=sns.color_palette("vlag", n_colors=2))

# Grafiğe başlık ekleme
plt.title('Distribution of Labels in the Dataset')

# Görseli göster
plt.show()


# In[8]:


# Metin uzunluklarının hesaplanması
text_lengths = data['cleaned_content'].apply(lambda x: len(x.split()))

# Maksimum ve ortalama dizi uzunluğu
max_length = text_lengths.max()
avg_length = text_lengths.mean()

print(f"Maksimum Uzunluk: {max_length}, Ortalama Uzunluk: {avg_length}")

# Kelime gömme boyutu genellikle bir hiperparametre olarak belirlenir.
# Örneğin, 100 veya 200 kullanabilirsiniz.
embedding_dim = 100


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertModel
from keras.layers import Input, Dense, concatenate
from keras.models import Model
import tensorflow as tf
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 'user_created' ve 'date' sütunlarını datetime formatına dönüştürme ve zaman dilimi bilgisi kaldırma
data['user_created'] = pd.to_datetime(data['user_created']).dt.tz_localize(None)
data['date'] = pd.to_datetime(data['date']).dt.tz_localize(None)

# Hesap yaşı hesaplama
data['account_age_days'] = (data['date'] - data['user_created']).dt.days

# Negatif veya NaN değerleri 0 ile değiştirme
data['account_age_days'] = data['account_age_days'].fillna(0).clip(lower=0)

from transformers import DistilBertTokenizer, TFDistilBertModel

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Modelinizi DistilBERT ile aynı şekilde kodlayıp kullanabilirsiniz
def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    
    for text in texts:
        text = tokenizer.encode_plus(text, max_length=max_len, padding='max_length',
                                     add_special_tokens=True, return_attention_mask=True, 
                                     truncation=True)
        
        all_tokens.append(text['input_ids'])
        all_masks.append(text['attention_mask'])
    
    return np.array(all_tokens), np.array(all_masks)

# Metin verilerini BERT için kodlama
max_len = 128  # Maksimum uzunluğu ayarlayın
input_ids, attention_masks = bert_encode(data['cleaned_content'], tokenizer, max_len=max_len)

# BERT'ten özelliklerin çıkarılması
bert_output = bert_model(input_ids, attention_mask=attention_masks)

# [CLS] tokenının gizli durumunu kullanarak özellikleri çıkar
bert_features = bert_output.last_hidden_state[:, 0, :].numpy()

# Veri setinizden kullanıcı profili özelliklerini seçme
user_profile_features = data[['user_followersCount', 'account_age_days']].values

from sklearn.impute import SimpleImputer
# NaN değerleri doldurma
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
user_profile_features = imputer.fit_transform(user_profile_features)

# Rastgele Orman modelini eğitme
rf_model = RandomForestClassifier()
rf_model.fit(user_profile_features, data['label'])
rf_predictions = rf_model.predict_proba(user_profile_features)[:, 1]

# BERT özellikleri için giriş katmanı
input_bert = Input(shape=(768,))  # BERT'ten çıkan özelliklerin boyutu
bert_output_layer = Dense(128, activation='relu')(input_bert)

# Kullanıcı profili verileri için giriş katmanı
input_rf = Input(shape=(1,))  # Rastgele Orman modelinin çıktısı
rf_output_layer = Dense(128, activation='relu')(input_rf)

# Hibrit modelin oluşturulması
combined = concatenate([bert_output_layer, rf_output_layer])
combined_output = Dense(64, activation='relu')(combined)
final_output = Dense(1, activation='sigmoid')(combined_output)

model = Model(inputs=[input_bert, input_rf], outputs=final_output)

# Modelin derlenmesi
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
X_train, X_test, y_train, y_test = train_test_split(np.hstack([bert_features, rf_predictions.reshape(-1, 1)]), data['label'], test_size=0.2)
history = model.fit([X_train[:, :768], X_train[:, 768:]], y_train, 
                    batch_size=32, epochs=10, 
                    validation_split=0.2, verbose=1)

# Eğitim ve doğrulama kayıplarını ve doğruluklarını çıkarma
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Modelin performansını değerlendirme
y_pred = model.predict([X_test[:, :768], X_test[:, 768:]])
y_pred = np.round(y_pred).astype(int)

# Metriklerin hesaplanması
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Karışıklık matrisinin hesaplanması
conf_matrix = confusion_matrix(y_test, y_pred)

# Define the function for plotting the custom confusion matrix with counts and percentages
def plot_confusion_matrix_with_counts_and_percentages(cm, ax):
    n = np.sum(cm)
    annot = np.array([["{}\n({:.2%})".format(cm[i, j], cm[i, j] / n) for j in range(cm.shape[1])] for i in range(cm.shape[0])])
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap='Greens', cbar=False, linewidths=1, linecolor='black')
    ax.set_xlabel('Predicted Values', fontsize=14)
    ax.set_ylabel('Actual Values', fontsize=14)
    ax.set_xticklabels(['Positive', 'Negative'], fontsize=12)
    ax.set_yticklabels(['Positive', 'Negative'], fontsize=12, rotation=0)
    ax.set_title('Confusion Matrix with Counts and Percentages', fontsize=16)

# Create a new figure for the updated confusion matrix
plt.figure(figsize=(10, 8))
ax = plt.gca()
plot_confusion_matrix_with_counts_and_percentages(conf_matrix, ax)
plt.tight_layout()
plt.show()

model.save('/Users/sulekaya/Downloads/2023/TEZ/kod/modelim.h5')  # Modeli H5 formatında kaydedin


# In[25]:


model.save('latest.h5')


# In[17]:


import matplotlib.pyplot as plt

# Eğitim ve doğrulama kayıp değerleri
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Eğitim ve doğrulama doğruluk değerleri
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Kayıp eğrisini çizdirme
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Doğruluk eğrisini çizdirme
plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.show()


# In[18]:


from sklearn.metrics import roc_curve, auc

# Modelin test verileri üzerindeki tahmin olasılıklarını al
y_pred_proba = model.predict([X_test[:, :768], X_test[:, 768:]]).ravel()

# ROC eğrisini ve AUC değerini hesaplama
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# ROC eğrisini çizdirme
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[21]:


from sklearn.model_selection import StratifiedKFold
# BERT ve Random Forest özelliklerini birleştirme
combined_features = np.hstack([bert_features, rf_predictions.reshape(-1, 1)])

# Etiketler
labels = data['label'].values

# Stratified K-Fold çapraz doğrulama nesnesi
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Her katlamada performans metriklerini saklamak için listeler
all_accuracies = []
all_precisions = []
all_recalls = []
all_f1_scores = []

# K-Fold çapraz doğrulama döngüsü
fold = 1
for train_index, test_index in skf.split(combined_features, labels):
    print(f"Starting Fold {fold}...")
    # Eğitim ve test veri setlerini ayırma
    X_train, X_test = combined_features[train_index], combined_features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Modelinizi tanımlama (önceki kod parçalarınızda olduğu gibi)
    # ...

    # Modeli eğitme
    history = model.fit([X_train[:, :768], X_train[:, 768:]], y_train, 
                        batch_size=32, epochs=10, 
                        validation_split=0.2, verbose=1)

    # Test veri seti üzerinde tahminler yapma
    y_pred = model.predict([X_test[:, :768], X_test[:, 768:]])
    y_pred = np.round(y_pred).astype(int)

    # Performans metriklerini hesaplama ve yazdırma
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Fold {fold} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}\n")

    all_accuracies.append(accuracy)
    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1_scores.append(f1)

    fold += 1

# Ortalama performans metriklerini yazdırma
print(f"Average Accuracy: {np.mean(all_accuracies)}")
print(f"Average Precision: {np.mean(all_precisions)}")
print(f"Average Recall: {np.mean(all_recalls)}")
print(f"Average F1 Score: {np.mean(all_f1_scores)}")


# In[15]:


# 'user_created' sütunundaki tarihleri datetime formatına dönüştürme
data['user_created_datetime'] = pd.to_datetime(data['user_created'])

# Irkçı ve ırkçı olmayan tweetlerin ayrılması
racist_tweets = data[data['label'] == 1]['user_created_datetime']
non_racist_tweets = data[data['label'] == 0]['user_created_datetime']


# In[23]:


from tensorflow.keras.utils import plot_model

# Modelinizi oluşturun
input_bert = Input(shape=(768,))  # BERT'ten çıkan özelliklerin boyutu
bert_output_layer = Dense(128, activation='relu')(input_bert)

input_rf = Input(shape=(1,))  # Rastgele Orman modelinin çıktısı
rf_output_layer = Dense(128, activation='relu')(input_rf)

combined = concatenate([bert_output_layer, rf_output_layer])
combined_output = Dense(64, activation='relu')(combined)
final_output = Dense(1, activation='sigmoid')(combined_output)

model = Model(inputs=[input_bert, input_rf], outputs=final_output)

# Önceden oluşturduğunuz modeli kullanarak görselleştirme yapın.
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

# Görselleştirilmiş modeli görüntüleyin
from IPython.display import Image
Image(filename='model_architecture.png')


# In[24]:


from tensorflow.keras.utils import plot_model

# Modelinizin oluşturulduğunu varsayalım ve 'model' değişkenine atandığını düşünelim.
plot_model(model, to_file='model_diagram.png', show_shapes=True, show_layer_names=True)


# In[16]:


from scipy.stats import mannwhitneyu
import numpy as np

# Unix zaman damgalarına dönüştürme
racist_timestamps = racist_tweets.values.astype(np.int64)
non_racist_timestamps = non_racist_tweets.values.astype(np.int64)

# Mann-Whitney U Testi
u_statistic, p_value = mannwhitneyu(racist_timestamps, non_racist_timestamps)
print("U Statistiği:", u_statistic, "P-Değeri:", p_value)

Kullanıcı profil analizi adımında, kullanıcıların Twitter üzerindeki etkileşimlerini ve davranışlarını incelemek için çeşitli görselleştirmeler yapacağım. Bu görselleştirmeler, takipçi sayıları, tweet beğeni sayıları, medya paylaşım sayıları ve hesapların oluşturulma yılları gibi özellikleri içerecek. 
# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns

# Görselleştirme için hazırlıklar
plt.style.use('ggplot')

# Takipçi Sayısının Dağılımı
plt.figure(figsize=(10, 6))
sns.histplot(data['user_followersCount'], bins=30, kde=True)
plt.title('Takipçi Sayısının Dağılımı')
plt.xlabel('Takipçi Sayısı')
plt.ylabel('Frekans')
plt.show()

# Takip Edilen Hesap Sayısının Dağılımı
plt.figure(figsize=(10, 6))
sns.histplot(data['friendsCount'], bins=30, kde=True)
plt.title('Takip Edilen Hesap Sayısının Dağılımı')
plt.xlabel('Takip Edilen Hesap Sayısı')
plt.ylabel('Frekans')
plt.show()

# Beğenilen Tweet Sayısının Dağılımı
plt.figure(figsize=(10, 6))
sns.histplot(data['favouritesCount'], bins=30, kde=True)
plt.title('Beğenilen Tweet Sayısının Dağılımı')
plt.xlabel('Beğenilen Tweet Sayısı')
plt.ylabel('Frekans')
plt.show()

# Yayınlanan Medya Sayısının Dağılımı
plt.figure(figsize=(10, 6))
sns.histplot(data['mediaCount'], bins=30, kde=True)
plt.title('Yayınlanan Medya Sayısının Dağılımı')
plt.xlabel('Medya Sayısı')
plt.ylabel('Frekans')
plt.show()

# Hesapların Oluşturulma Yılı
data['user_created_year'] = pd.to_datetime(data['user_created']).dt.year
plt.figure(figsize=(10, 6))
sns.countplot(x='user_created_year', data=data)
plt.title('Hesapların Oluşturulma Yılına Göre Dağılımı')
plt.xlabel('Yıl')
plt.ylabel('Hesap Sayısı')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


#Analizler


# In[29]:


#b. Duygu Analizi:
#Duygu analizi için TextBlob veya VADER gibi kütüphaneler kullanılabilir. Bu analiz, metinlerin genel duygu tonunu belirlemek için yapılır.

from textblob import TextBlob

# Duygu puanını hesaplama ve veriye ekleme
data['sentiment'] = data['cleaned_content'].apply(lambda x: TextBlob(x).sentiment.polarity)


# In[30]:


#c. Konu Modelleme:
#Konu modelleme için gensim kütüphanesini kullanabiliriz. LDA (Latent Dirichlet Allocation) modeli, metinlerdeki gizli konuları belirlemek için kullanılabilir.

from gensim import corpora
from gensim.models.ldamodel import LdaModel
import gensim

# Metinlerin işlenmesi
texts = [[word for word in document.lower().split() if word not in turkish_stopwords] for document in data['cleaned_content']]

# Sözlük ve corpus oluşturma
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# LDA modelinin eğitilmesi
lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)

# Konuların yazdırılması
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)


# In[31]:


#3. Etkileşim Analizi:
#Etkileşim ve popülerlik analizi, beğeni, retweet ve yanıt sayılarına bakarak yapılabilir. Bu analiz, ırkçı ve yabancı düşmanı tweetlerin diğer tweetlere göre daha fazla veya daha az etkileşim alıp almadığını belirlemek için kullanılır.
# Etkileşim sayılarının hesaplanması
data['total_interactions'] = data['likeCount'] + data['retweetCount'] + data['replyCount'] + data['quoteCount']

# Etkileşim sayılarının label'a göre analizi
interaction_analysis = data.groupby('label')['total_interactions'].mean()
print(interaction_analysis)


# In[33]:


#4. Kullanıcı Profili Analizi:
#Kullanıcı profili analizi, kullanıcıların davranışlarını ve profil özelliklerini incelemek için yapılır. Anomali tespiti, normalden sapma gösteren hesapları belirlemek için kullanılabilir.
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

# Eksik değerleri doldurmak için imputer oluşturma
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# 'user_followersCount' ve 'account_age_days' sütunları için eksik değerleri doldurma
data_imputed = imputer.fit_transform(data[['user_followersCount', 'account_age_days']])

# Anomali tespiti için modelin eğitilmesi
iso_forest = IsolationForest(n_estimators=100)
data['anomaly_score'] = iso_forest.fit_predict(data_imputed)

# Anomali puanlarının incelemesi
anomalies = data[data['anomaly_score'] == -1]
print("Anomali sayısı:", anomalies.shape[0])


# In[34]:


anomalies = data[data['anomaly_score'] == -1]
print(anomalies[['content', 'user_followersCount', 'account_age_days', 'likeCount', 'retweetCount']])


# In[35]:


# Anomali olarak işaretlenen verilerin seçilmesi
anomalies = data[data['anomaly_score'] == -1]

# Anomali olarak işaretlenen hesapların yaşı, takipçi sayıları ve etkileşim düzeylerinin incelenmesi
anomalies_descriptive_stats = {
    'account_age_days': anomalies['account_age_days'].describe(),
    'user_followersCount': anomalies['user_followersCount'].describe(),
    'total_interactions': anomalies['likeCount'] + anomalies['retweetCount'] + anomalies['replyCount'] + anomalies['quoteCount']
}

# Yüzdelik hesaplamaları için
anomalies_descriptive_stats['total_interactions'] = anomalies_descriptive_stats['total_interactions'].describe()

anomalies_descriptive_stats


# In[ ]:




