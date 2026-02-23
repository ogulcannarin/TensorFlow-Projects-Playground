# 🧠 İleri Seviye Derin Öğrenme Mimarileri

Bu klasör, klasik sinir ağlarından başlayarak günümüzün en modern AI modellerine (Transformer, GNN) kadar uzanan geniş bir mimari yelpazesini uygulamalı olarak sunar.

## 🏗️ Mimari Kütüphanesi

### 👗 Fashion-MNIST (Giriş)
- **Dosya:** `00-Fashion-MNIST-Giris.py`
- **Konu:** Standart YSA (Yapay Sinir Ağı) ile 10 farklı kıyafet türünü tanıma.

### 👁️ CNN (Convolutional Neural Networks)
- **Dosya:** `01-CNN-MNIST.py`
- **İşlem:** `Conv2D` ve `MaxPooling2D` katmanları ile görüntüdeki kenarları ve desenleri yakalama.
- **Avantaj:** Standart ağlara göre çok daha az parametreyle görsel veride yüksek başarı.

### ⏳ RNN (Recurrent Neural Networks)
- **Dosya:** `02-RNN-Sayi-Tahmini.py`
- **İşlem:** Ardışık (sequential) verileri işlemek için `SimpleRNN` kullanımı.
- **Kullanım Alanı:** Sayı dizisi tahmini, metin üretme.

### ⚡ Transformer (Multi-Head Attention)
- **Dosya:** `03-Transformer-Mimarisi.py`
- **Teknik Detay:** 
  - `MultiHeadAttention` katmanı ile her verinin diğerleriyle ilişkisini hesaplar.
  - `LayerNormalization` ve `Residual Connections` (Artık Bağlantılar) ile eğitim kararlılığı.
  - GPT ve BERT gibi modellerin nasıl çalıştığını anlamak için tasarlanmış fonksiyonel yapı.

### 🕸️ GNN (Graph Neural Networks)
- **Dosya:** `04-GNN-Graf-Aglar.py`
- **Konu:** Birbirine bağlı veriler (sosyal ağlar, molekül yapıları) üzerinde derin öğrenme.

## 🔧 Teknik Notlar
- Modellerde `Functional API` ve `Sequential` olmak üzere iki farklı Keras yaklaşımı da örneklendirilmiştir.
- Her script, ilgili mimarinin teorik prensiplerini kod üzerinde yorum satırlarıyla açıklar.

---
*Mimariyi anlamak, yapay zekanın mantığını çözmektir.*
