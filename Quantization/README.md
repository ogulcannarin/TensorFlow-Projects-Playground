# 🐱🐶 Kedi ve Köpek Görüntü Sınıflandırma & TFLite Nicemleme (Quantization)

Bu proje, TensorFlow ve Keras kullanarak kedi ve köpek görüntülerini sınıflandıran bir Evrişimli Sinir Ağı (CNN) modelinin eğitilmesini ve ardından bu modelin mobil cihazlarda veya kısıtlı donanımlarda (IoT vb.) verimli bir şekilde çalışabilmesi için **8-bit TFLite** formatına dönüştürülmesini (Quantization) sağlar.

## 🚀 Projenin Amacı

Yapay zeka modelleri genellikle yüksek boyutlu ve hesaplama maliyeti yüksek modellerdir. Bu projenin temel amacı, eğitilmiş bir derin öğrenme modelinin (`.h5`) başarımından çok fazla ödün vermeden, boyutunu küçülterek ve hızını artırarak TensorFlow Lite formatına (`.tflite`) nasıl dönüştürülebileceğini (8-bit Integer Quantization) göstermektir.

## 📁 Proje Yapısı ve Dosyalar

Proje birbirini takip eden 3 temel aşamadan oluşmaktadır:

*   **`egitim.py`**: 
    *   Veri setini (`dogs-vs-cats`) `cat` ve `dog` alt klasörlerine ayırarak TensorFlow'un anlayacağı formata getirir.
    *   Verileri normalize eder ve bir CNN mimarisi oluşturup modeli eğitir.
    *   Eğitim sonucunda `kedi_kopek_ham_model.h5` adlı ham derin öğrenme modelini üretir.

*   **`quantize.py`**:
    *   Eğitilmiş olan `.h5` modelini sisteme yükler.
    *   TFLiteConverter kullanarak, temsili bir veri seti (representative dataset) yardımıyla modeli **8-bit (uint8)** olarak nicemler (Quantization).
    *   Cihaz üzerinde çok daha hafif, düşük bellek tüketen ve hızlı çalışacak olan `kedi_kopek_8bit.tflite` modelini oluşturur.

*   **`dogruluk_test.py`**:
    *   Oluşturulan `.tflite` modelini belleğe yükleyerek girdi (input) ve çıktı (output) veri tiplerini (dtype) test eder.
    *   Modelin başarılı bir şekilde 8-bit (`uint8`) formatına dönüştüğünü doğrular.

## 🛠️ Kullanılan Teknolojiler

*   **Python 3.x**
*   **TensorFlow & Keras** (Model Eğitimi ve Sınıflandırma)
*   **TensorFlow Lite (TFLite)** (Model Optimizasyonu ve Nicemleme)
*   **NumPy** (Veri İşleme)
*   **OS & Shutil** (Dosya ve Klasör Yönetimi)

## ⚙️ Kurulum ve Çalıştırma

### 1. Gereksinimleri Yükleyin
Projenin çalışması için Python bilgisayarınızda kurulu olmalı ve terminal üzerinden gerekli kütüphaneleri yüklemelisiniz:
```bash
pip install tensorflow numpy
```

### 2. Veri Setini Hazırlayın
Proje dizininde `dogs-vs-cats/train/train` şeklinde bir klasör yolu bulunmalıdır. İçerisinde karışık halde kedi ve köpek fotoğrafları (`cat.1.jpg`, `dog.1.jpg` vb.) yer almalıdır. (Eğitim kodu bu klasörleri otomatik olarak düzenleyecektir.)

### 3. Aşamaları Sırasıyla Çalıştırın

**Adım 1: Modeli Eğitin**
Aşağıdaki komutu çalıştırarak verilerin düzenlenmesini ve modelin eğitilmesini sağlayın. İşlem bittiğinde `kedi_kopek_ham_model.h5` dosyası oluşacaktır. (Bu işlem bilgisayarınızın donanımına göre birkaç dakika sürebilir.)
```bash
python egitim.py
```

**Adım 2: Modeli Küçültün (Quantization)**
Eğitilen modeli donanım dostu TFLite formatına çevirmek için aşağıdaki komutu çalıştırın. Bu işlem sonucunda `kedi_kopek_8bit.tflite` dosyası elde edilecektir.
```bash
python quantize.py
```

**Adım 3: Test Edin**
Oluşturulan modelin başarıyla 8-bit'e dönüştüğünü görmek için doğruluk testini çalıştırın. Terminalde giriş ve çıkış tiplerinin `<class 'numpy.uint8'>` olduğunu görmelisiniz.
```bash
python dogruluk_test.py
```

## 📝 Lisans
Bu proje eğitim amacıyla hazırlanmış açık kaynaklı bir projedir. Geliştirmeye ve kopyalamaya açıktır.
