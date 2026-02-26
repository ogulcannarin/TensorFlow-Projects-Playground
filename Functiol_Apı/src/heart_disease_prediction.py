import pandas as pd
import numpy as np 
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import os

# ==========================================
# 1. VERİ YÜKLEME VE HAZIRLIK
# ==========================================
print("Veri yükleniyor...")
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'heart_disease_data.csv')
df = pd.read_csv(data_path)
y = df['HeartDiseaseorAttack']
X = df.drop('HeartDiseaseorAttack', axis=1)

# Veriyi Eğitim (%80) ve Test (%20) olarak bölme
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sayıları Ölçeklendirme (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# ==========================================
# 2. MODEL MİMARİSİ (Functional API)
# ==========================================
def build_advanced_model(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    
    # Temel Giriş Katmanı
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Residual (Artık) Bağlantı Bloğu
    shortcut = x 
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    
    # Giriş ile derin katmanları topluyoruz (Skip Connection)
    x = layers.add([x, shortcut]) 
    x = layers.Activation('relu')(x)
    
    # Çıkış Katmanı
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

model = build_advanced_model(X_train.shape[1])
model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy', tf.keras.metrics.Recall()]
)

# ==========================================
# 3. EĞİTİM
# ==========================================
print("\nModel Eğitiliyor... Lütfen Bekleyin.")
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)

# ==========================================
# 4. HASSASİYET AYARI VE ANALİZ
# ==========================================
print("\nModel Değerlendiriliyor...")
y_pred_probs = model.predict(X_test)

# Kritik Eşik Değeri (Recall artırmak için %15 seçtik)
esik_degeri = 0.15 
y_pred_hassas = (y_pred_probs > esik_degeri).astype(int)

# Karmaşıklık Matrisini Çizdir
cm = confusion_matrix(y_test, y_pred_hassas)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.xlabel(f'Tahmin Edilen (Eşik: {esik_degeri})')
plt.ylabel('Gerçek Durum')
plt.title('Hassas Model Analizi (Confusion Matrix)')
plt.show()

print(f"\n--- SINIFLANDIRMA RAPORU (Eşik: {esik_degeri}) ---")
print(classification_report(y_test, y_pred_hassas))

# ==========================================
# 5. ETKİLEŞİMLİ RİSK ÖLÇER FONKSİYONU
# ==========================================
def manuel_risk_testi():
    print("\n" + "="*40)
    print("      KALP SAĞLIĞI RİSK SİMÜLASYONU")
    print("="*40)
    print("Lütfen aşağıdaki soruları yanıtlayın (Evet: 1, Hayır: 0)")
    
    try:
        # Veri setindeki orijinal 21 sütun sırasıyla girişleri alıyoruz
        sorular = [
            "Yüksek Tansiyonunuz var mı?", "Yüksek Kolesterolünüz var mı?",
            "Son 5 yılda kolesterol kontrolü yaptırdınız mı?", "Vücut Kitle Endeksiniz (BMI)?",
            "Sigara kullanıyor musunuz?", "Hiç felç geçirdiniz mi?",
            "Diyabet durumunuz? (0:Yok, 1:Pre-diyabet, 2:Diyabet)", "Fiziksel aktivite yapıyor musunuz?",
            "Meyve tüketiyor musunuz?", "Sebze tüketiyor musunuz?",
            "Ağır alkol tüketimi var mı?", "Sağlık sigortanız var mı?",
            "Maliyetten dolayı doktora gidemediğiniz oldu mu?", "Genel sağlık durumunuz? (1:Çok İyi, 5:Kötü)",
            "Ruh sağlığınızın kötü olduğu gün sayısı (0-30)", "Fiziksel sağlığınızın kötü olduğu gün sayısı (0-30)",
            "Yürürken zorlanıyor musunuz?", "Cinsiyetiniz? (1:Erkek, 0:Kadın)",
            "Yaş kategoriniz? (1:18-24 ... 13:80+)", "Eğitim seviyeniz (1-6)", "Yıllık gelir seviyeniz (1-8)"
        ]
        
        user_inputs = []
        for soru in sorular:
            val = float(input(f"{soru}: "))
            user_inputs.append(val)

        # Tahmin süreci
        girdi_dizisi = np.array([user_inputs])
        girdi_olcekli = scaler.transform(girdi_dizisi)
        tahmin = model.predict(girdi_olcekli, verbose=0)[0][0]
        
        print("\n" + "-"*30)
        print(f"Hastalık Olasılığı: %{tahmin * 100:.2f}")
        
        if tahmin > esik_degeri:
            print(f"SONUÇ: YÜKSEK RİSK (Eşik: %{esik_degeri*100:.0f})")
            print("Öneri: Bir kardiyoloğa görünmeniz tavsiye edilir.")
        else:
            print("SONUÇ: DÜŞÜK RİSK")
            print("Öneri: Sağlıklı yaşam tarzına devam edin!")
        print("-"*30)

    except Exception as e:
        print(f"Hata oluştu: {e}. Lütfen sayısal değerler girin.")

# Uygulamayı çalıştır
manuel_risk_testi()