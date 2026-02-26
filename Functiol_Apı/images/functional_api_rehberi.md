# Keras Functional API: Matematiksel ve Grafiksel Bir Rehber

Derin öğrenme modelleri tasarlarken genellikle ardışık (sequential) katmanlar kullanırız. Ancak bazı modern mimariler (ResNet, Inception, vb.) birden fazla giriş/çıkış, dallanma (branching) veya katman atlama (skip connection) gerektirir. İşte bu noktada **Keras Functional API** devreye girer.

Bu rehberde Functional API'nin mantığını, arkasında yatan graf teorisini ve matematiksel işlemleri inceleyeceğiz.

---

## 1. Functional API Nedir?

Keras Functional API, modelleri ardışık bir liste yerine **Yönlü Asiklik Grafik (Directed Acyclic Graph - DAG)** olarak tanımlamanıza olanak tanıyan bir arayüzdür. 

Daha basit bir ifadeyle:
- **Sequential API**, katmanların peş peşe geldiği düz bir yoldur.
- **Functional API**, yolların ayrılabildiği, köprülerin (skip connections) kurulabildiği karmaşık bir otoyol ağıdır.

### Örnek Sözdizimi:
```python
import tensorflow as tf
from tensorflow.keras import layers

inputs = tf.keras.Input(shape=(32,))
x = layers.Dense(64, activation='relu')(inputs)
outputs = layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```
Burada dikkat edilmesi gereken nokta şudur: **Katmanları çağırırken onları fonksiyonlar gibi davranarak doğrudan önceki tensor'a uyguluyoruz: `(inputs)` veya `(x)`.**

---

## 2. Hesaplama Grafikleri (Computational Graphs)

Derin öğrenme çerçeveleri matematiği hesaplarken arka planda **Hesaplama Grafikleri (Computational Graphs)** oluşturur. Functional API bu grafikleri doğrudan sizin kontrolünüze bırakır.

### Grafiğin Bileşenleri
1. **Düğüm (Node):** Her matematiksel işlem (Dense katmanı, Aktivasyon, Toplama vs.) grafikte bir düğümdür.
2. **Kenar (Edge / Tensor):** Bir düğümün çıktısı olan çok boyutlu matrisler (Tensorlar), başka bir düğüme akan veriyi (kenarı) temsil eder.

**Örnek Graf Yapısı:**
```
[Giriş: Input] --> [Düğüm: Dense (64)] --> [Kenar: Tensor] --> [Düğüm: Dense (32)] --> [Çıkış]
```

Functional API ile tasarlanan bir yapay sinir ağı, matematiksel işlemlerin bir zinciri değil, matematiksel fonksiyonların birbirinin içine geçmesidir (Fonksiyonel Kompozisyon).

---

## 3. Arkasındaki Matematik

Functional API kullanırken arka planda klasik matris kuralı geçerlidir. Ancak yapının esnek olması sayesinde matematiği çok daha zengin hale getirebiliriz.

### A. Matris Çarpımı (İleri Besleme / Feedforward Linear Math)

Bir `Dense(units)` katmanı kullanıldığında arkada şu işlem gerçekleşir:
$$ Y = f(X \cdot W + b) $$

- $X$: Giriş tensorü / Matrisi
- $W$: Ağırlık matrisi (Weights)
- $b$: Sapma vektörü (Bias)
- $f$: Aktivasyon fonksiyonu (ReLU, Sigmoid vb.)

### B. Aktivasyon Fonksiyonları Matematiği

Giriş ile çıktı arasındaki doğrusal olmayan özelliklerin (non-linearity) öğrenilmesini sağlar.

- **ReLU (Rectified Linear Unit):** 
  $$ f(x) = \max(0, x) $$
  *(Negatif değerleri sıfırlar, pozitifleri aynen geçirir.)*

- **Sigmoid (Çıkış katmanı için):**
  $$ f(x) = \frac{1}{1 + e^{-x}} $$
  *(Sonuçları 0 ile 1 arasına sıkıştırarak bir "Olasılık" üretir. Kalp hastalığı risk analizinde kullanıldığı gibi.)*

### C. Artık Bağlantılar (Residual / Skip Connections) Matematiği

Bu projedeki `heart_disease_prediction.py` dosyasında da kullanıldığı gibi Functional API'nin en güçlü yanı katmanları birbirine "toplama" seçeneği sunmasıdır:
```python
x = layers.add([x, shortcut])
```
Matematiksel olarak **ResNet (Residual Networks)** mimarisinin temeli olan bu işlem şu formülle gösterilir:
$$ Y = f(x) + x $$

Burada:
- $x$: Önceki katmandan gelen ham giriş ("shortcut").
- $f(x)$: Aktivasyon ve dönüşüm uygulanan çıktı.

**Bunun amacı nedir?**
Derin ağlarda geriye yayılım (Backpropagation) sırasında yaşanan *gradyan kaybolmasını (Vanishing Gradient)* önler. Eğer ağ hiçbir şey öğrenemezse bile en azından doğrudan giriş bilgisi ($+ x$) çıkışa iletilir. Türev zincir kuralında $\frac{d}{dx} (f(x) + x) = f'(x) + 1$ olduğu için gradyan hiçbir zaman tamamen sıfır olmaz.

---

## 4. Projemizdeki Mimari: Ne İfade Ediyor?

Eğer `heart_disease_prediction.py` içerisindeki modele bakarsanızFunctional API gücünü göreceksiniz:

1. **Input Katmanı:** Hastalara ait 21 farklı parametre ağa girer: `shape=(21,)`
2. **Dense Bloğu:** `Dense(64)` -> İşlemler: $y_1 = ReLU(W_1X + b_1)$
3. **Shortcut Kaydı:** `shortcut = x` -> Mevcut durum cebimizde! Bellekte bir "kenar(edge)" yedeği alıyoruz.
4. **Derinleşme (Dropout ve Dense):** İki farklı gizli katmandan geçiyor. Matematik karmaşıklaşıyor ($f_{derin}$).
5. **Düğüm Birleşimi (Add):** `layers.add([x, shortcut])` algoritmasıyla yeni özellikler ile ilk kaydettiğimiz orijinal özellikler toplanıyor: 
 $$ Yeni Çıkış = f_{derin}(X) + shortcut $$
6. **Çıkış Düğümü:** `Dense(1, activation='sigmoid')` -> 0 ile 1 arası olasılık skoru hesaplanıyor.

Özetle, Keras Functional API yalnızca bir "kod yazım tarzı" değil, aynı zamanda veriyi grafik teorisindeki ağlarla dilediğiniz gibi şekillendirip matematiksel sınırları aşmanıza yarayan güçlü bir hesaplama tasarımıdır.
