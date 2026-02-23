# 🔢 TensorFlow Temel İşlemler

Bu klasör, TensorFlow kütüphanesinin çekirdek mekanizmalarını ve makine öğrenmesinin en temel taşlarını uygulamalı olarak içerir. Başlangıç seviyesinden orta seviyeye doğru bir öğrenme eğrisi sunar.

## 📁 Dosya Detayları ve Teknik İçerik

### 1. TensorFlow Temelleri (`00-Giris-Tensor-Basics.py`)
- **İçerik:** `tf.constant`, `tf.Variable` ve Tensör matematiksel işlemleri.
- **Kazanım:** TensorFlow'un veriyi nasıl temsil ettiğini ve hesaplama grafiklerini anlama.

### 2. Basit Ev Fiyat Tahmini (`01-Ev-Fiyat-Tahmini-Basit.py`)
- **Senaryo:** Oda sayısına göre ev fiyatı tahmini (Tek değişkenli model).
- **Mimari:** Tek bir `Dense(1)` katmanı.
- **Matematik:** `y = wx + b` formülünün sinir ağı tarafından öğrenilmesi.

### 3. TensorFlow Ameliyathanesi (`01-Temel-Islemler.py`)
- **İçerik:** Matris çarpımları, transpoz işlemleri ve veri tipi (casting) dönüşümleri.
- **Önem:** Derin öğrenme modellerinin arka planındaki lineer cebir işlemlerini kavramak.

### 4. Lineer Regresyon Uygulaması (`02-Lineer-Regresyon.py`)
- **İçerik:** Gürültülü (noisy) verilerden bir doğruyu tahmin etme.
- **Teknik:** Keras `Sequential` API kullanımı. `Mean Squared Error (MSE)` kaybı ile modelin optimize edilmesi.

### 5. Gelişmiş Lojistik Regresyon (`03-Lojistik-Regresyon.py`)
- **Veri Seti:** Ünlü **MNIST** (El yazısı rakamlar) veri seti.
- **Mimari:** 
  - `Flatten`: 28x28 görüntüleri 784 boyutlu vektörlere çevirir.
  - `Dense(128)`: Gizli katman (ReLU aktivasyonu).
  - `Dropout(0.2)`: Aşırı öğrenmeyi (Overfitting) engellemek için nöron kapatma tekniği.
  - `Softmax`: Çoklu sınıflandırma (0-9 arası rakamlar).

## 🚀 Nasıl Başlanır?
Bu klasördeki scriptler herhangi bir harici veri dosyasına ihtiyaç duymaz (MNIST internetten otomatik çekilir).

```bash
python 03-Lojistik-Regresyon.py
```

---
*Bu bölümü tamamladığınızda, TensorFlow'un veriyi nasıl işlediğini ve basit bir sinir ağının nasıl kurulduğunu öğrenmiş olacaksınız.*
