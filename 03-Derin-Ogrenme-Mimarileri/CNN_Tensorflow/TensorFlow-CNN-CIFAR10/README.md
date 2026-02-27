# 🖼️ CIFAR-10 Veri Kümesi ile CNN Sınıflandırma

Bu proje kapsamında, **TensorFlow** ve **Keras** kütüphaneleri kullanılarak **CIFAR-10** veri seti üzerinde bir nesne sınıflandırma modeli geliştirilmiştir. CIFAR-10 veri kümesi; içerisinde uçak, araba, kuş, kedi, geyik, köpek, kurbağa, at, gemi ve kamyon olmak üzere 10 farklı sınıf barındıran zengin bir veri kümesidir.

## 🚀 Projenin Amacı ve Özeti

Derin öğrenme alanında yaygın olarak bilinen kaynaklardan biri olan CIFAR-10 veri setini okuyup, uygun ölçeklendirmeden (normalization) geçirdikten sonra bir Evrişimli Sinir Ağı (CNN - Convolutional Neural Network) modeli tasarlamak ve modeli eğitmektir.

### 🧱 Model Mimarisi
Model, ardışık (Sequential) bir API kullanılarak tasarlanmıştır. İçerisinde aşağıdaki katmanları bulundurur:
* **Conv2D ve MaxPooling2D:** Görüntüden öznitelik (özellik) haritalarını ve sınırları belirlemek için kullanılmıştır. (Aktivasyon: `relu`)
* **Flatten:** Matriks şeklindeki öznitelikleri vektör haline getirmek için kullanılmıştır.
* **Dense:** Sonuçları sınıflamak için kullanılmış tamamen bağlı katmanlardır.

## 📊 Eğitim ve Değerlendirme

- **Optimizasyon:** `adam` algoritması kullanılmıştır.
- **Kayıp Fonksiyonu (Loss):** `SparseCategoricalCrossentropy`
- **Eğitim Süreci:** Toplam **10 epoch** kullanılarak çalıştırılmış olup, her epoch sonunda modelin validasyon (doğrulama) başarı oranı kaydedilmiştir.

### Sonuçlar

Eğitim sürecine ait veri örnekleri/model eğitim grafikleri aşağıdaki gibidir:

![Model Eğitim / Veri Çıktısı](Ekran%20görüntüsü%202026-02-27%20122332.png)

## 💻 Çalıştırma
Projeyi çalıştırmak için bağımlılıkların (`tensorflow`, `matplotlib`) kurulu olduğundan emin olun ve ardından `orn.py` dosyasını çalıştırın:
```bash
python orn.py
```
