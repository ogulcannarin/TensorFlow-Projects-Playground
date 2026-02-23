import tensorflow as tf
import numpy as np

# Fashion MNIST veri setini yükle
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Görselleri normalize et (0-1 aralığına)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Veri şekillerini inceleme
print("Eğitim verisi şekli:", train_images.shape)
print("Eğitim etiketleri şekli:", train_labels.shape)

# Modeli tanımla
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Modeli derle
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model özeti
model.summary()

# Modeli eğit
model.fit(train_images, train_labels, epochs=5)

# Test setinde değerlendir
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)

# Birkaç tahmin yap
predictions = model.predict(test_images)
print("İlk test görüntüsünün tahmini:", np.argmax(predictions[0]))
print("Gerçek etiket:", test_labels[0])