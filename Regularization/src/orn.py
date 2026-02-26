import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. VERİ YÜKLEME
# ==========================================

print("Veri yükleniyor...\n")

df = pd.read_csv("../data/heart_disease_data.csv")

print("İlk 5 satır:")
print(df.head())

print("\nDataset Bilgisi:")
print(df.info())

print("\nDataset Boyutu:")
print(df.shape)

print("\nSütun İsimleri:")
print(df.columns)

# ==========================================
# 2. HEDEF DEĞİŞKEN ANALİZİ
# ==========================================

print("\nKalp Hastalığı Sayıları:")

print(df["HeartDiseaseorAttack"].value_counts())

print("\nOranlar:")

print(df["HeartDiseaseorAttack"].value_counts(normalize=True))


# ==========================================
# 3. TARGET GRAFİĞİ
# ==========================================

plt.figure(figsize=(6,4))

sns.countplot(x="HeartDiseaseorAttack", data=df)

plt.title("Kalp Hastalığı Dağılımı")

plt.show()


# ==========================================
# 4. KORELASYON MATRİSİ
# ==========================================

print("\nKorelasyon Hesaplanıyor...")

plt.figure(figsize=(12,8))

sns.heatmap(df.corr())

plt.title("Feature Correlation")

plt.show()


# ==========================================
# 5. BMI ANALİZİ
# ==========================================

print("\nOrtalama BMI:", df["BMI"].mean())

sns.histplot(df["BMI"], bins=40)

plt.title("BMI Distribution")

plt.show()


# ==========================================
# 6. AGE ANALİZİ
# ==========================================

print("\nOrtalama Yaş Kategorisi:", df["Age"].mean())

sns.histplot(df["Age"], bins=13)

plt.title("Age Distribution")

plt.show()


# ==========================================
# 7. AGE vs HASTALIK
# ==========================================

print("\nYaşa Göre Kalp Hastalığı Ortalaması")

print(df.groupby("HeartDiseaseorAttack")["Age"].mean())


sns.boxplot(x="HeartDiseaseorAttack", y="Age", data=df)

plt.title("Age vs Heart Disease")

plt.show()


# ==========================================
# 8. HIGHBP ANALİZİ
# ==========================================

print("\nTansiyon Ortalamaları:")

print(df.groupby("HeartDiseaseorAttack")["HighBP"].mean())


sns.countplot(x="HighBP", hue="HeartDiseaseorAttack", data=df)

plt.title("HighBP vs HeartDisease")

plt.show()


# ==========================================
# 9. FEATURE ÖNEM ANALİZİ
# ==========================================

print("\nFeature Önem Analizi:")

corr = df.corr()

target_corr = corr["HeartDiseaseorAttack"].sort_values(ascending=False)

print(target_corr)


target_corr.drop("HeartDiseaseorAttack").plot(kind="bar")

plt.title("Feature Importance")

plt.show()
# ==========================================
# 10. VERİ ÖN İŞLEME (PREPROCESSING)
# ==========================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Bağımsız değişkenler (X) ve Hedef değişken (y)
X = df.drop("HeartDiseaseorAttack", axis=1)
y = df["HeartDiseaseorAttack"]

# Veriyi Eğitim (%80) ve Test (%20) olarak bölüyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizasyon: Derin öğrenmede verilerin aynı ölçekte (0-1 arası gibi) olması hayati önem taşır
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nVeri hazırlama tamamlandı. Model kuruluyor...")

# ==========================================
# 11. MODEL KURMA (REGULARIZATION İLE)
# ==========================================
from tensorflow.keras import regularizers

model = models.Sequential([
    # Katman 1: L2 Regularization (Ağırlıkların aşırı büyümesini engeller)
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],),
                 kernel_regularizer=regularizers.l2(0.01)), 
    
    # Dropout: Eğitim sırasında nöronların %30'unu rastgele kapatır (Ezberlemeyi önler)
    layers.Dropout(0.3),
    
    # Katman 2
    layers.Dense(32, activation='relu', 
                 kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.2),
    
    # Çıkış Katmanı (İkili sınıflandırma olduğu için Sigmoid kullanılır)
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# ==========================================
# 12. MODELİ EĞİTME
# ==========================================
# validation_split: Eğitim verisinin %20'sini eğitim sırasında test için ayırır
history = model.fit(X_train, y_train, 
                    epochs=30, 
                    batch_size=64, 
                    validation_split=0.2, 
                    verbose=1)

# ==========================================
# 13. SONUÇLARI GÖRSELLEŞTİRME
# ==========================================
plt.figure(figsize=(12, 5))

# Kayıp (Loss) Grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kayıp (Loss) Analizi')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Doğruluk (Accuracy) Grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Eğitim Başarısı')
plt.plot(history.history['val_accuracy'], label='Doğrulama Başarısı')
plt.title('Model Doğruluk (Accuracy) Analizi')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# ==========================================
# 14. TEST SETİ ÜZERİNDE DEĞERLENDİRME
# ==========================================
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Seti Doğruluğu: {test_acc:.4f}")
# ==========================================
# 15. REGULARIZATION TESTİ (DAHA SERT CEZALANDIRMA)
# ==========================================
print("\nSert Regularization ile model tekrar eğitiliyor...")

model_test = models.Sequential([
    # L2 değerini 0.01'den 0.1'e çıkardık (Ağırlıkları daha çok baskılar)
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],),
                 kernel_regularizer=regularizers.l2(0.1)), 
    
    # Dropout oranını 0.3'ten 0.5'e çıkardık (Nöronların yarısı her adımda kapanacak)
    layers.Dropout(0.5),
    
    layers.Dense(32, activation='relu', 
                 kernel_regularizer=regularizers.l2(0.1)),
    layers.Dropout(0.5),
    
    layers.Dense(1, activation='sigmoid')
])

model_test.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Eğitimi başlatıyoruz
history_test = model_test.fit(X_train, y_train, 
                              epochs=30, 
                              batch_size=64, 
                              validation_split=0.2, 
                              verbose=1)

# ==========================================
# 16. YENİ SONUÇLARI GÖRSELLEŞTİRME
# ==========================================
plt.figure(figsize=(12, 5))

# Kayıp (Loss) Karşılaştırması
plt.subplot(1, 2, 1)
plt.plot(history_test.history['loss'], label='Yeni Eğitim Kaybı', color='red')
plt.plot(history_test.history['val_loss'], label='Yeni Doğrulama Kaybı', color='green')
plt.title('Sert Regularization: Kayıp Grafiği')
plt.legend()

# Doğruluk (Accuracy) Karşılaştırması
plt.subplot(1, 2, 2)
plt.plot(history_test.history['accuracy'], label='Yeni Eğitim Başarısı', color='red')
plt.plot(history_test.history['val_accuracy'], label='Yeni Doğrulama Başarısı', color='green')
plt.title('Sert Regularization: Doğruluk Grafiği')
plt.legend()

plt.show()
# ==========================================
# 17. OPTİMUM MODEL (DENGELİ REGULARIZATION)
# ==========================================
from tensorflow.keras.optimizers import Adam

model_final = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],),
                 kernel_regularizer=regularizers.l2(0.005)), # Daha hafif ceza
    layers.Dropout(0.2), # Daha az nöron kapatma
    
    layers.Dense(32, activation='relu', 
                 kernel_regularizer=regularizers.l2(0.005)),
    layers.Dropout(0.2),
    
    layers.Dense(1, activation='sigmoid')
])

# Öğrenme hızını biraz düşürüyoruz (Varsayılan 0.001'dir, biz 0.0005 yapalım)
optimizer = Adam(learning_rate=0.0005)

model_final.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

history_final = model_final.fit(X_train, y_train, 
                                epochs=50, # Daha yavaş öğrendiği için epoch'u artırdık
                                batch_size=64, 
                                validation_split=0.2, 
                                verbose=1)