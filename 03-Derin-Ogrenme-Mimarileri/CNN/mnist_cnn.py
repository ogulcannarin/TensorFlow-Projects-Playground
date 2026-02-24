import os
# Uyarı kirliliğini engellemek için (importlardan önce ekliyoruz)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# 1. VERİ SETİNİ YÜKLEME
print("Veri seti indiriliyor ve hazırlanıyor...")
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Veriyi normalize etme (Piksel değerlerini 0-255 arasından 0-1 arasına çekiyoruz)
# Bu adım modelin çok daha hızlı öğrenmesini sağlar.
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# 2. MODEL MİMARİSİ
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # 10 farklı rakam sınıfı için
])

# 3. MODELİ DERLEME
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. EĞİTİMİ BAŞLATMA
print("\nEğitim başlıyor... (Bu işlem işlemcinize göre 1-2 dakika sürebilir)\n")
model.fit(train_images, train_labels, epochs=3, batch_size=64)

# 5. TEST ETME
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest Başarısı: %{test_acc*100:.2f}')