# 🐾 TensorFlow Kedi-Köpek Sınıflandırma Projesi

Bu proje, Kaggle üzerindeki devasa bir veri setini kullanarak, bir bilgisayarın kedi ve köpek resimlerini birbirinden nasıl ayırt edebileceğini gösteren uçtan uca bir görüntü işleme (Computer Vision) uygulamasıdır.

## 🏗️ Model Mimarisi: CNN (Convolutional Neural Network)

Model, görüntüleri katman katman analiz ederek önce kenarları, sonra şekilleri ve nihayetinde nesneleri (kulak, burun vb.) tanımayı öğrenir:

1.  **Rescaling Katmanı:** Görüntü piksellerini [0, 255] aralığından [0, 1] aralığına normalize eder.
2.  **Conv2D & MaxPooling (3 Katman):**
    -   32, 64 ve 128 filtreli evrişim katmanları.
    -   Her katman sonrası veriyi küçülterek önemli özellikleri öne çıkaran MaxPooling.
3.  **Flatten Katmanı:** 2D matrisleri 1D vektöre dönüştürerek sınıflandırma kısmına hazırlar.
4.  **Dense Katmanı (512 Nöron):** Yakalanan özellikleri derinlemesine analiz eder.
5.  **Output Katmanı (Sigmoid):** Tek bir çıktı üretir (0: Kedi, 1: Köpek).

## 📊 Veri Hazırlama Süreci (Pipeline)

- **Otomatik Düzenleme:** `main.py` çalıştığında, 25.000 resmi isimlerine göre `cat/` ve `dog/` klasörlerine otomatik olarak dağıtır.
- **Validasyon Bölümü:** Verinin %20'si eğitim sırasında modeli test etmek için otomatik olarak ayrılır (`validation_split=0.2`).
- **Verim:** `image_dataset_from_directory` fonksiyonu ile veriler diskten verimli bir şekilde okunur, belleği yormaz.

## 📈 Eğitim Sonuçları
Eğitim bittiğinde, aşağıdaki değerleri içeren bir grafik oluşturulur:
- **Eğitim Doğruluğu vs Doğrulama Doğruluğu**
- **Eğitim Kaybı vs Doğrulama Kaybı**

Bu grafikler, modelin veriyi ezberleyip ezberlemediğini (Overfitting) kontrol etmenizi sağlar.

## 🔮 Tahmin Yapma (Prediction)
`cats_vs_dogs_model.h5` dosyası oluştuktan sonra, herhangi bir resmi test etmek için:
1. Resmi bu klasöre `test_image.jpg` adıyla kaydedin.
2. `predict.py` dosyasını çalıştırın.
3. Ekranda tahmin sonucu ve güven oranı belirecektir.

---
*Bilgisayarların dünyayı bizim gibi görmesi için ilk adım...* 🐶🐱
