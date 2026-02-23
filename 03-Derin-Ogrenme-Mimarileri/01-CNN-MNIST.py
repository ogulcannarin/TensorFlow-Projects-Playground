import tensorflow as tf
from tensorflow.keras import layers, models

# 1. MNIST Verisini Yükle
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Veriyi Şekillendir (CNN 4 boyutlu veri bekler: Adet, Yükseklik, Genişlik, Kanal)
# MNIST siyah-beyaz olduğu için kanal sayısı 1'dir.
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

# 3. CNN Modelini Oluştur
model = models.Sequential([
    # İlk Evrişim Katmanı: 32 farklı 3x3 boyutunda filtre kullanır
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)), # Resmin boyutunu küçültür (önemli yerleri tutar)
    
    # İkinci Evrişim Katmanı: Daha karmaşık desenleri yakalar
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Klasik Sinir Ağı Katmanlarına Geçiş
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 4. Modeli Derle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Eğit
print("CNN Modeli eğitiliyor (Bu işlem CPU'da biraz vakit alabilir)...")
model.fit(x_train, y_train, epochs=5)

# 6. Test Et
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nCNN Test Doğruluğu: %{test_acc*100:.2f}')