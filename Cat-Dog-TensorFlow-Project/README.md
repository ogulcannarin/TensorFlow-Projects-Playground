# TensorFlow Kedi-Köpek Sınıflandırma Projesi 🐱🐶

Bu proje, TensorFlow ve Keras kütüphanelerini kullanarak bir **Evrişimli Sinir Ağı (CNN)** modelini eğitir. Model, kendisine verilen bir resmin kedi mi yoksa köpek mi olduğunu yüksek bir doğruluk oranıyla tahmin etmeyi öğrenir.

Bu depo, modeli eğitmek (`main.py`) ve eğitilmiş modeli kullanarak tahmin yapmak (`predict.py`) için gereken tüm kodları içerir.

## 🚀 Kullanılan Teknolojiler

* **Python 3.10+**
* **TensorFlow (Keras):** Modelin oluşturulması, eğitilmesi ve yüklenmesi için.
* **Matplotlib:** Eğitim sonuçlarının (başarı ve kayıp grafikleri) görselleştirilmesi için.
* **OS Modülü:** Veri setinin (25.000 resim) otomatik olarak `cat` ve `dog` klasörlerine ayrılması için.
* **Kaggle:** 800MB'lık orijinal veri setinin kaynağı.

---

## 🏁 Nasıl Kullanılır?

Bu projeyi bilgisayarınızda çalıştırmak için 3 ana adımı takip etmeniz gerekmektedir.

### Adım 1: Kurulum

1.  Bu depoyu bilgisayarınıza klonlayın veya indirin.
2.  Gerekli Python kütüphanelerini `pip` kullanarak yükleyin:
    ```bash
    pip install tensorflow matplotlib h5py
    ```

### Adım 2: Veri Setini Hazırlama ve Modeli Eğitme

Modeli eğitmek için `main.py` script'i kullanılır. Ancak bu script'in çalışması için orijinal veri setine ihtiyacı vardır.

1.  **Veri Setini İndirin:**
    [Kaggle: Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) yarışma sayfasından `train.zip` (yaklaşık 812 MB) dosyasını indirin.

2.  **Veriyi Çıkartın:**
    İndirdiğiniz `train.zip` dosyasını, bu proje klasörünün içine çıkartın. İşlem bittiğinde, proje klasörünüzde `train` adında (içinde 25.000 resim olan) bir alt klasör oluşmalıdır.

    Klasör yapısı şöyle görünmelidir:
    ```
    tensorflow-cat-dog-project/
    |-- main.py
    |-- predict.py
    |-- .gitignore
    |-- README.md
    |-- train/
        |-- cat.0.jpg
        |-- dog.0.jpg
        |-- ... (binlerce resim)
    ```

3.  **Modeli Eğitin:**
    Hazırlıklar tamamsa, terminalden `main.py` dosyasını çalıştırın:
    ```bash
    python main.py
    ```

Bu script, `train` klasöründeki resimleri otomatik olarak `train/cat` ve `train/dog` klasörlerine ayıracak, modeli 3 tur (epoch) boyunca eğitecek ve işlem bittiğinde `cats_vs_dogs_model.h5` adında eğitilmiş bir model dosyası kaydedecektir.

*(Not: Bu eğitim işlemi, bilgisayarınızın işlemci hızına bağlı olarak 15-20 dakika sürebilir.)*

### Adım 3: Tahmin Yapma (Eğlenceli Kısım!)

Modeli bir kez eğittikten sonra, `main.py`'yi tekrar çalıştırmanıza gerek yoktur. `predict.py` script'i, kaydedilen `.h5` modelini kullanarak saniyeler içinde tahmin yapabilir.

1.  İnternetten rastgele bir kedi veya köpek resmi bulun.
2.  Resmi proje klasörüne (`predict.py`'nin yanına) indirin ve adını **`test_image.jpg`** olarak değiştirin.
3.  Aşağıdaki komutu çalıştırın:
    ```bash
    python predict.py
    ```

