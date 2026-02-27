# 👗 Fashion-MNIST ile Giyim Tarzı Sınıflandırma

Bu proje kapsamında **Fashion-MNIST** veri seti kullanılarak, derin öğrenme ile kıyafet sınıflandırması (T-Shirt, Pantolon, Kazak, Elbise, Ayakkabı, Çanta vb.) yapan bir yapay sinir ağı tasarlanmıştır.

## 💡 Proje Hakkında

Fashion-MNIST, standart olan El Yazısı rakamları (MNIST) veri setine benzer yapısıyla, ancak bir boyutta daha karmaşık olduğu için bilgisayarlı görü projelerinde yaygınca kullanılan bir veri setidir. Toplamda 10 farklı kıyafet türünü barındırır. Görüntüler `28x28` piksel boyutundadır ve tek kanallıdır (siyah/beyaz).

### ⚙️ Çalışma Mantığı
* Önce eğitim ve test verisi ayrılıp belleğe yüklenir ve sonrasında kolay işlem yapılabilmesi için her bir piksel değeri `[0, 1]` aralığına getirilecek şekilde (255'e bölünerek) ölçeklenir.
* Gelen görüntü `28x28` formatından, klasik Yapay Sinir Ağlarına verilebilmesi adına düzleştirilir (`Flatten`).
* Ardından gizli katmanlar aracılığıyla (`Dense` ve `relu` aktivasyonları) özellikleri elde edilir ve en son olarak tahminde bulunulur.

## 📊 Sonuçlar ve Görsel Tahminler

Aşağıda modelin kıyafet tahmini konusunda nasıl bir iş çıkardığını görebilirsiniz:

### Eğitim Verilerinden Örnekler
Modelin eğitime başlamadan önce gördüğü veri örnekleri ve görselleştirmeler:

| Giriş Örneği | Veri Kümesi Taraması |
| :---: | :---: |
| ![Görsel 1](Ekran%20görüntüsü%202026-02-27%20120207.png) | ![Görsel 2](Ekran%20görüntüsü%202026-02-27%20120228.png) |

### Model Tahminleri ve Değerlendirme
Modelin her bir test girdisi için yaptığı tahminler aşağıda listelenmiştir. 
Kırmızı çubuk: *Hatalı Tahminleri*, Mavi çubuk: *Doğru Tahminlerin olasılık gücünü* temsil eder. 

| Tahmin Analizi 1 | Tahmin Analizi 2 |
| :---: | :---: |
| ![Görsel 3](Ekran%20görüntüsü%202026-02-27%20120320.png) | ![Görsel 4](Ekran%20görüntüsü%202026-02-27%20120338.png) |
| ![Görsel 5](Ekran%20görüntüsü%202026-02-27%20120356.png) | ![Görsel 6](Ekran%20görüntüsü%202026-02-27%20120434.png) |

> *Not: Sağ tarafta yer alan grafikler, sınıflandırma ihtimallerinin ürün etiketlerine göre olasılık dağılımını ifade eder.*

## 💻 Kullanım
Bağımlılıklar olarak NumPy, Matplotlib ve TensorFlow kurulu olmalıdır. Kodu test etmek ve görselleştirilen ekranlara ulaşmak için:
```bash
python orn.py
```
