# 🧠 Derin Öğrenme ve GAN (Üretken Çekişmeli Ağlar) Temelleri

Bu rehber, sinir ağlarının temel çalışma prensiplerini ve özellikle **GAN (Generative Adversarial Networks - Üretken Çekişmeli Ağlar)** mimarisinin nasıl oluşturulduğunu, eğitildiğini ve kullanıldığını açıklamak amacıyla hazırlanmıştır. Aşağıda yer alan görseller, derin öğrenme dünyasındaki zorlukları ve çözümleri adım adım özetlemektedir.

---

## 🏗️ 1. Temel Sinir Ağları ve Zorluklar

Derin öğrenme modelleri oluşturulurken verilerin ağ içinden nasıl geçtiğini ve dağılımların modeli nasıl etkilediğini anlamak çok önemlidir. GAN mimarisini oluşturmadan önce bu temellerin sağlam atılmış olması gerekir.

### Aktivasyonlar (Activations)
![Activations](./images/Ekran%20görüntüsü_28-2-2026_221731_www.coursera.org.jpeg)

**Ne Anlatıyor?** 
Bir sinir ağının her bir nöronundaki (düğümündeki) temel matematiksel işlemi gösterir. Önceki katmandan gelen girdiler ($a^{[l-1]}$), ağırlıklar ($W$) ile çarpılıp sapma (bias, $b$) ile toplanarak $z$ değeri elde edilir. Daha sonra bu $z$ değeri bir **aktivasyon fonksiyonundan** ($g$) geçirilerek (örneğin ReLU, Sigmoid) nörona doğrusal olmayan özellikler ($a$) kazandırılır. Görselde ayrıca, modelin "kürk rengi" veya "boyut" gibi özelliklere nasıl odaklanabileceği betimlenmiştir.
**Kullanım Alanı:** 
Tüm derin öğrenme mimarilerinin temel yapı taşıdır. Ağın karmaşık desenleri ve fonksiyonları öğrenebilmesini sağlar.

### Ortak Değişken Kayması (Covariate Shift)
![Covariate Shift](./images/Ekran%20görüntüsü_28-2-2026_231037_www.coursera.org.jpeg)

**Ne Anlatıyor?** 
Giriş verilerinin (veya gizli katmanlardaki verilerin) dağılımındaki değişimi ifade eder. Görselde $x_2$ verisinin dağılımının (siyah grafikten beyaz grafiğe) nasıl kaydığını ve bu kaymanın maliyet fonksiyonunu (cost function) nasıl bozduğunu görüyoruz.
**Kullanım Alanı:** 
Covariate shift problemi, eğitimin yavaşlamasına ve dengesizleşmesine yol açar. **Batch Normalization (Toplu Normalizasyon)** gibi teknikler, bu etkiyi azaltmak ve özellikle GAN'lar gibi hassas, eğitilmesi zor modelleri dengeli bir şekilde eğitebilmek için yaygın olarak kullanılır.

---

## ⚔️ 2. GAN Mimarisine Giriş

GAN (Üretken Çekişmeli Ağlar), birbirine karşı kıyasıya rekabet eden iki ana sinir ağından oluşur: **Üretici (Generator)** ve **Ayrıştırıcı (Discriminator)**.

### Ayrıştırıcı (Discriminator)
![Discriminator](./images/Ekran%20görüntüsü%202026-02-28%20143319.png)

**Ne Anlatıyor?** 
Sistemin "dedektif" veya "eleştirmen" kısmıdır. Görsel, bir resmin sinir ağına girip belirli olasılıklar dahilinde (Örn: %45 Kedi, %45 Köpek, %10 Kuş) sınıflandırılmasını göstermektedir.
**Kullanım Alanı:** 
GAN mimarisinde Discriminator, kendisine verilen bir görselin gerçekveri setinden mi geldiğini yoksa üretici tarafından yaratılmış "sahte" bir veri mi olduğunu ayırt etmeye çalışır. Standart resim sınıflandırma problemlerinde kullanılan modeller (CNN vb.) discriminator görevi görebilir.

### Üretici (Generator / Neural Networks)
![Generator](./images/Ekran%20görüntüsü%202026-02-28%20143838.png)

**Ne Anlatıyor?** 
Sistemin "kalpazanı" veya "sanatkârı" diyebiliriz. Rastgele sayılardan oluşan bir **"Gürültü" (Noise)** vektörünün karmaşık bir sinir ağından geçerek yepyeni, sentetik bir görsele (örnekte tüysüz bir kedi fotoğrafına) dönüştürülmesini resmeder.
**Kullanım Alanı:** 
Tamamen yoktan taze ve orijinal veri üretmek için kullanılır. Metinden görsel üretmek, yaşlandırma efektleri yapmak, deepfake teknolojileri veya eksik olan veriyi tamamlamak gibi yapay zekanın "üretken" (generative) tarafını temsil eder.

---

## 🏋️‍♂️ 3. GAN Modellerinin Eğitimi

Bir GAN modelinin başarılı ve dengeli olabilmesi için her iki ağın da oyun teorisindeki Minimax (rekabet) mantığıyla sırayla ve doğru şekilde eğitilmesi gerekir.

### Ayrıştırıcının Eğitimi (Training Discriminator)
![Training Discriminator](./images/Ekran%20görüntüsü_28-2-2026_22124_www.coursera.org.jpeg)

**Ne Anlatıyor?** 
Discriminator eğitimi sırasında hem gerçek veriler ($X$) hem de üreticiden gelen sahte veriler ($\hat{X}$) ayrıştırıcıya beslenir. Amaç, discriminator'ın gerçeğe 1, sahteye 0 diyebilmesini öğretmektir. Maliyet hesaplanır ve geriye yayılım ile **sadece discriminator'ın parametreleri ($\theta_d$)** güncellenir.
**Kullanım Alanı:** 
Modelin sahtekarlıkları ne kadar iyi yakalayabildiğini geliştirdiği aşamadır. Güçlü ve sürekli güncellenen bir discriminator, onu kandırabilmek için daha kaliteli görseller üretmesi gereken generator'ı da sürekli olarak sınırlarını zorlamaya iter.

### Üreticinin Eğitimi (Training Generator)
![Training Generator](./images/Ekran%20görüntüsü_28-2-2026_221241_www.coursera.org.jpeg)

**Ne Anlatıyor?** 
Bu aşamada ayrıştırıcının öğrenmesi durdurulur ve parametreleri ($\theta_d$) dondurulur (görseldeki çarpı işareti bunu ifade eder). Gürültüden üretilen sahte örnekler doğrudan discriminator'a gönderilir, ancak bu kez ağ kandırılmaya çalışıldığı için maliyet (Cost) hesaplanırken **sahte görsellerin etiketleri sistemde bilerek "gerçek"miş (1) gibi** kabul edilir. Discriminator bunları sahte olarak fark ederse büyük bir ceza/maliyet üretilir ve bu ceza kullanılarak geriye yayılım ile **sadece üreticinin parametreleri ($\theta_g$)** güncellenir.
**Kullanım Alanı:** 
Üreticinin, ayrıştırıcıyı (yani algoritmayı) "kandırmayı" öğrendiği yerdir. Bu eğitim döngüsü (gerçek-sahte çatışması) sürekli devam eder ve gerçeğinden ayırt edilemeyen çıktılar üretene dek sürer.

---

## 🎨 4. Sonuç ve Kullanım (Sampling)

### Veri Üretimi / Örnekleme (Sampling)
![Sampling](./images/Ekran%20görüntüsü%202026-02-28%20144144.png)

**Ne Anlatıyor?** 
GAN modelinin eğitimi tatmin edici bir seviyeye geldiğinde, ayrıştırıcı (discriminator) devreden çıkarılır ve atılır. Artık elimizde sadece eğitilmiş, parametreleri ($\theta$) oturmuş **kaydedilmiş bir Üretici (Saved Generator)** bulunmaktadır. Bu ağa farklı rastgele gürültü (noise) vektörleri verdikçe yepyeni, eşsiz ama eğitim veri setindeki nesneye benzeyen yüksek kaliteli örnekler (görselde farklı köpek fotoğrafları) oluşturulur.
**Kullanım Alanı:** 
Eğitilmiş bir yapay zeka modelinin "üretim ve son kullanım" aşamasıdır. Tasarımcılar için ilham karakterleri üretmek, video oyunları için devasa haritalar veya sınırsız varyantta objeler (araba, ağaç, yüz vs.) hazırlamak için kullanılır.
