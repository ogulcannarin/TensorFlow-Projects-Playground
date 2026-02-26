import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# ==========================================
# 1. VERİ HAZIRLIĞI
# ==========================================
# Train verisini yükle (Dosya adının doğruluğundan emin ol)
df = pd.read_csv("train.csv")

# Sayısal sütunları seç ve Hedef (SalePrice) değişkenini ayır
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
y = df['SalePrice'].values
X = df[num_cols].drop('SalePrice', axis=1).values

# Eksik değerleri ortalama ile doldur
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Veriyi normalize et (SGD için bu adım hayati önem taşır)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Eğitim ve Doğrulama setlerine ayır
X_train, X_val, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. MODEL MİMARİSİ
# ==========================================
def build_model():
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.2), # Overfitting'i engellemek için eklendi
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Fiyat tahmini (Regression) olduğu için tek çıktı
    ])
    return model

# ==========================================
# 3. OPTIMIZER KARŞILAŞTIRMASI
# ==========================================

# --- SGD Deneyi ---
model_sgd = build_model()
optimizer_sgd = tf.keras.optimizers.SGD(learning_rate=0.001) # Çok büyük sayılarda düşük LR daha güvenlidir
model_sgd.compile(optimizer=optimizer_sgd, loss='mse', metrics=['mae'])

print("\nSGD ile Eğitim Başlıyor...")
history_sgd = model_sgd.fit(
    X_train, y_train,
    validation_data=(X_val, y_test),
    epochs=50,
    batch_size=32,
    verbose=0 # Ekranı çok kirletmemesi için 0 yaptık
)

# --- Adam Deneyi ---
model_adam = build_model()
optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.01)
model_adam.compile(optimizer=optimizer_adam, loss='mse', metrics=['mae'])

print("Adam ile Eğitim Başlıyor...")
history_adam = model_adam.fit(
    X_train, y_train,
    validation_data=(X_val, y_test),
    epochs=50,
    batch_size=32,
    verbose=0
)

# ==========================================
# 4. SONUÇLARIN GÖRSELLEŞTİRİLMESİ
# ==========================================
plt.figure(figsize=(12, 6))

# Kayıpları (Loss) çizdir
plt.plot(history_sgd.history['val_loss'], label='SGD Val Loss (Doğrulama)', color='red', linestyle='--')
plt.plot(history_adam.history['val_loss'], label='Adam Val Loss (Doğrulama)', color='blue', linestyle='--')

plt.title("Optimizer Karşılaştırması: Adam vs SGD")
plt.xlabel("Epoch")
plt.ylabel("MSE (Hata Oranı)")
plt.yscale('log') # Hatalar çok büyükse grafiği logaritmik görmek daha net sonuç verir
plt.legend()
plt.grid(True)
plt.show()

print("\nİşlem tamamlandı! Grafikte hangi çizginin daha hızlı düştüğüne bakabilirsin.")