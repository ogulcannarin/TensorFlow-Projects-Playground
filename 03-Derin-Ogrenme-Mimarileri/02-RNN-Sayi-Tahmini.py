import tensorflow as tf
import numpy as np

# 1. Veri Hazırlama
# Elimizde şöyle bir kural olsun: Her sayı bir öncekinden 10 fazla.
# X: [ [10, 20, 30], [20, 30, 40], [30, 40, 50] ] -> Giriş dizisi
# Y: [ 40, 50, 60 ] -> Tahmin edilmesi gereken bir sonraki sayı

X = np.array([
    [10, 20, 30],
    [20, 30, 40],
    [30, 40, 50],
    [40, 50, 60]
], dtype=np.float32)

Y = np.array([40, 50, 60, 70], dtype=np.float32)

# RNN modelleri veriyi (Örnek Sayısı, Zaman Adımı, Özellik Sayısı) formatında bekler.
X = X.reshape((X.shape[0], X.shape[1], 1))

# 2. RNN Modelini Oluştur
model = tf.keras.Sequential([
    # SimpleRNN katmanı: 50 nöronlu bir hafıza alanı
    tf.keras.layers.SimpleRNN(50, activation='relu', input_shape=(3, 1)),
    tf.keras.layers.Dense(1) # Tek bir sayı tahmin edeceğiz
])

# 3. Modeli Derle
model.compile(optimizer='adam', loss='mean_squared_error')

# 4. Eğit
print("RNN Sayı dizisini öğreniyor...")
model.fit(X, Y, epochs=1000, verbose=0) # 1000 tur eğit ama ekrana basma

# 5. Tahmin Et
# Modelden 50, 60, 70 dizisinden sonra ne geleceğini soraşım (Beklenen: 80)
test_input = np.array([50, 60, 70], dtype=np.float32).reshape((1, 3, 1))
tahmin = model.predict(test_input)

print(f"\n[50, 60, 70] dizisinden sonra gelen sayı tahmini: {tahmin[0][0]:.2f}")