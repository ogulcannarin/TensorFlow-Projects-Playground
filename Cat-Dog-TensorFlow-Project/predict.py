import tensorflow as tf
import numpy as np
import os

# 1. Sabitler (Constants)
MODEL_PATH = 'cats_vs_dogs_model.h5'
IMAGE_TO_TEST = 'test_image.jpg' # Test edilecek resmin adı
IMG_WIDTH = 160
IMG_HEIGHT = 160

# 2. Resmi modele uygun hale getiren fonksiyon
def prepare_image(image_path):
    """Verilen yoldaki resmi yükler ve modelin anlayacağı formata getirir."""
    # Resmi 160x160 boyutunda yükle
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    
    # Resmi bir numpy dizisine çevir (shape: 160, 160, 3)
    img_array = tf.keras.utils.img_to_array(img)
    
    # Modeller tek bir resim yerine 'batch' (grup) bekler.
    # Bu yüzden resmin boyutunu (1, 160, 160, 3) şeklinde genişletiyoruz.
    img_array = np.expand_dims(img_array, axis=0)
    
    # Not: 1./255'e bölmüyoruz, çünkü modelimizin ilk katmanı
    # 'Rescaling' katmanı ve bu işi zaten yapıyor.
    return img_array

# 3. Ana Tahmin Fonksiyonu
def predict_image():
    # Modelin ve resmin var olup olmadığını kontrol et
    if not os.path.exists(MODEL_PATH):
        print(f"HATA: Model dosyası '{MODEL_PATH}' bulunamadı.")
        print("Lütfen önce 'main.py' dosyasını çalıştırarak modeli eğittiğinizden ve kaydettiğinizden emin olun.")
        return

    if not os.path.exists(IMAGE_TO_TEST):
        print(f"HATA: Resim dosyası '{IMAGE_TO_TEST}' bulunamadı.")
        print(f"Lütfen internetten bir kedi/köpek resmi indirin ve adını '{IMAGE_TO_TEST}' olarak bu klasöre kaydedin.")
        return

    # Modeli yükle (1-2 saniye sürer)
    print(f"Eğitilmiş model '{MODEL_PATH}' yükleniyor...")
    try:
        # 'compile=False' eklemek, özel kayıp fonksiyonu olmadığı için yüklemeyi hızlandırır
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        # Modeli tekrar derlememiz gerekiyor (sadece tahmin için)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        print("Model başarıyla yüklendi.")
        
    except Exception as e:
        print(f"Model yüklenirken bir hata oluştu: {e}")
        print("Lütfen 'pip install h5py' komutunu çalıştırdığınızdan emin olun.")
        return

    # Resmi hazırla
    prepared_img = prepare_image(IMAGE_TO_TEST)

    # Tahmini yap
    print(f"'{IMAGE_TO_TEST}' için tahmin yapılıyor...")
    
    # model.predict, her zaman bir grup (batch) sonuç döner
    prediction = model.predict(prepared_img)
    
    # Sonucu yorumla
    # prediction, [[0.99]] (köpek) veya [[0.01]] (kedi) gibi bir değer döner
    score = prediction[0][0]

    print("\n--- TAHMİN SONUCU ---")
    if score < 0.5:
        # Sınıf 0 (cat)
        print(f"Bu resim %{100 * (1 - score):.2f} ihtimalle bir KEDİ.")
    else:
        # Sınıf 1 (dog)
        print(f"Bu resim %{100 * score:.2f} ihtimalle bir KÖPEK.")
    print(f"(Ham skor: {score})") # 0'a yakınsa kedi, 1'e yakınsa köpek


# Script'i çalıştır
if __name__ == "__main__":
    predict_image()