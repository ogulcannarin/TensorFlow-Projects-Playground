import tensorflow as tf

# 1. MNIST Verisini Yükle
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Veriyi Normalize Et (0-1 arasına çekiyoruz)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. Modeli Oluştur (Gizli Katman Eklendi)
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  
  # Gizli Katman: 128 nöronlu bu katman modelin "zekasını" artırır
  tf.keras.layers.Dense(128, activation='relu'), 
  
  # Dropout: Eğitim sırasında nöronların %20'sini rastgele kapatır. 
  # Bu sayede model ezberlemek yerine (overfitting) genellemeyi öğrenir.
  tf.keras.layers.Dropout(0.2), 
  
  tf.keras.layers.Dense(10, activation='softmax')
])

# 4. Modeli Derle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Eğit
print("Gelişmiş model eğitiliyor (Hidden Layer devrede)...")
model.fit(x_train, y_train, epochs=5)

# 6. Test Et
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nGelişmiş Model Test Doğruluğu: %{test_acc*100:.2f}')