import tensorflow as tf
import numpy as np

# 1. Veri Hazırlama
# Transformer'lar (Örnek Sayısı, Dizi Uzunluğu, Özellik Sayısı) formatında veri bekler.
# 100 farklı örnek, her biri 5 elemanlı bir dizi ve her eleman 8 farklı özellik içeriyor.
X = np.random.rand(100, 5, 8).astype(np.float32) 
Y = np.random.rand(100, 1).astype(np.float32)

# 2. Transformer Bloğu Fonksiyonu
# Bu yapı, ChatGPT gibi dev modellerin temel yapı taşıdır.
def transformer_block(inputs):
    # Self-Attention (Öz-Dikkat): Dizideki her parça, diğer tüm parçalara aynı anda bakar.
    # num_heads=2: Farklı "bakış açıları" oluşturur.
    attention = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=2)(inputs, inputs)
    
    # Add & Norm: Girdiyi sonuçla toplayıp normalize eder (Eğitimi hızlandırır ve sabitler).
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + inputs)
    
    # Feed Forward kısmı: Bilgiyi işleyen klasik katman
    ff_output = tf.keras.layers.Dense(8, activation="relu")(x)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(ff_output + x)

# 3. Modeli İnşa Etme (Functional API kullanarak)
inputs = tf.keras.Input(shape=(5, 8))
x = transformer_block(inputs)
x = tf.keras.layers.GlobalAveragePooling1D()(x) # Diziden gelen veriyi özetle
outputs = tf.keras.layers.Dense(1)(x) # Sonuç tahmini

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# 4. Modeli Derle
model.compile(optimizer='adam', loss='mse')

# 5. Model Özetini Yazdır
print("--- Transformer Model Özeti ---")
model.summary()

# 6. Küçük Bir Eğitim Yapalım
print("\nModel eğitiliyor...")
model.fit(X, Y, epochs=10, verbose=1)

print("\nTransformer mimarisi başarıyla kuruldu ve çalıştırıldı!")