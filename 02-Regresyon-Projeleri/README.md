# 📉 Regresyon Projeleri: Veri Analizi ve Tahminleme

Bu klasörde, ham verinin bir makine öğrenmesi modeline aktarılmadan önce geçmesi gereken tüm kritik adımlar (Temizleme, Görselleştirme, Ölçeklendirme) gerçek projeler üzerinden gösterilmektedir.

## 📊 Öne Çıkan Proje: Mercedes Fiyat Tahmini

Bu çalışma, bir veri bilimcinin günlük hayatta karşılaştığı veri kirliliği ile nasıl başa çıkılacağını özetler.

### 🛠️ İzlenen Veri Bilimi İş Akışı
1.  **Keşifsel Veri Analizi (EDA):** `Seaborn` ile fiyat dağılımı, yıl ve kilometre arasındaki ilişkiler incelendi.
2.  **Veri Temizleme (Outlier Removal):** Fiyatı aşırı yüksek olan %1'lik dilim silinerek modelin sapma yapması engellendi. (131 araç elendi).
3.  **Özellik Mühendisliği:** String değer içeren kolonlar (şanzıman tipi gibi) modelin hata vermemesi için çıkarıldı.
4.  **Veri Ölçeklendirme:** `MinMaxScaler` kullanılarak tüm özellikler 0 ile 1 arasına getirildi.
5.  **Derin Sinir Ağı Mimarisi:**
    - 4 Gizli Katman (Her biri 12 nöronlu, ReLU aktivasyonlu).
    - Çıkış Katmanı (1 nöron, Lineer).
6.  **Gelişmiş Eğitim Teknikleri:**
    - `EarlyStopping`: Modelin test kaybı (val_loss) artmaya başladığında eğitimi otomatik durdurur (Patience: 15).
    - `TensorBoard`: `logs/` klasörüne kaydedilen loglar sayesinde eğitimi tarayıcıda izleme imkanı.

### 📁 Klasör Yapısı
- `01-Bisiklet-Fiyatlari.py`: ANN ile temel regresyon örneği.
- `02-Mercedes-Fiyat-Tahmini.py`: Tam kapsamlı veri analizi ve tahmin projesi.
- `../datasets/`: Kullanılan `.xlsx` ve `.csv` veri dosyaları.

## 📈 Model Değerlendirme
Proje sonunda **MAE (Mean Absolute Error)** ve **MSE (Mean Squared Error)** metrikleri hesaplanır. Ayrıca "Gerçek vs Tahmin" grafiği ile modelin doğruluğu görsel olarak teyit edilir.

---
*Verinin fısıldadığını duymak için önce onu temizlemek gerekir.*
