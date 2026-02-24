import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.preprocessing import sequence

# 1. PARAMETRELER (Ayarlar)
max_features = 10000  # En sık kullanılan 10.000 kelimeye odaklan
maxlen = 100         # Her yorumun ilk 100 kelimesini oku (Hafıza sınırı)

# 2. VERİ SETİNİ YÜKLE
print("IMDB Film yorumları indiriliyor...")
(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=max_features)

# Cümle boylarını eşitle (Kısa cümlelerin sonuna 0 ekler, uzunları keser)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# 3. RNN MODELİNİ İNŞA ET
model = models.Sequential([
    # Embedding: Kelimeleri birbirleriyle olan anlam ilişkilerine göre vektörleştirir
    layers.Embedding(max_features, 32),
    
    # SimpleRNN: İşte hafıza burada devreye giriyor!
    # Kelimeleri sırayla okuyup bir önceki kelimeyi hatırlar.
    layers.SimpleRNN(32), 
    
    # Karar Katmanı: Sonucu 0 (Kötü) veya 1 (İyi) olarak belirler
    layers.Dense(1, activation='sigmoid')
])

# 4. MODELİ DERLE
model.compile(optimizer='rmsprop', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# 5. EĞİTİMİ BAŞLAT
print("\nYapay zeka yorumları okumaya başlıyor...\n")
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

# 6. TEST ET
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nModelin Test Başarısı: %{acc*100:.2f}")