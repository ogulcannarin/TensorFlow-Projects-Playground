import tensorflow as tf
import matplotlib.pyplot as plt
import os
import ssl

# SSL sertifika sorunlarını es geçmek için
ssl._create_default_https_context = ssl._create_unverified_context

print(f"TensorFlow Sürümü: {tf.__version__}")

# 1. Veri setimizin bulunduğu klasörün yolu
base_dir = r'C:\Users\ogulc\Downloads\tensoflow'
train_dir = os.path.join(base_dir, 'train')

# train klasörünün var olup olmadığını kontrol et
if not os.path.exists(train_dir):
    print(f"HATA: '{train_dir}' klasörü bulunamadı.")
    print(f"Lütfen Kaggle'dan 'train.zip' dosyasını indirip '{base_dir}' klasörüne çıkarttığınızdan emin olun.")
else:
    print(f"Veri klasörü bulundu: {train_dir}")

    # Görüntüleri hangi boyuta getireceğimizi tanımlayalım
    IMG_SIZE = (160, 160)
    BATCH_SIZE = 32

    # 2. ---- RESİMLERİ AYIKLAMA KODU ----
    cat_dir = os.path.join(train_dir, 'cat')
    dog_dir = os.path.join(train_dir, 'dog')

    if not os.path.exists(cat_dir) or not os.path.exists(dog_dir):
        print("'cat' ve 'dog' alt klasörleri oluşturuluyor...")
        os.makedirs(cat_dir, exist_ok=True)
        os.makedirs(dog_dir, exist_ok=True)
        
        fnames = os.listdir(train_dir)
        print(f"Taşınacak dosyalar aranıyor... (Toplam {len(fnames)} dosya kontrol edilecek)")
        
        file_count = 0
        for fname in fnames:
            if fname.endswith('.jpg'): 
                src = os.path.join(train_dir, fname)
                dst = ""
                if fname.startswith('cat'):
                    dst = os.path.join(cat_dir, fname)
                elif fname.startswith('dog'):
                    dst = os.path.join(dog_dir, fname)
                
                if dst:
                    try:
                        os.rename(src, dst)
                        file_count += 1
                    except FileExistsError:
                        pass 
        print(f"{file_count} adet resim 'cat' ve 'dog' klasörlerine başarıyla ayrıldı.")
    else:
        print("'cat' ve 'dog' klasörleri zaten mevcut. Ayıklama adımı atlanıyor.")

    # 3. VERİYİ YÜKLEME
    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
          train_dir,
          validation_split=0.2,
          subset="training",
          seed=123,
          image_size=IMG_SIZE,
          batch_size=BATCH_SIZE
        )

        validation_ds = tf.keras.utils.image_dataset_from_directory(
          train_dir,
          validation_split=0.2,
          subset="validation",
          seed=123,
          image_size=IMG_SIZE,
          batch_size=BATCH_SIZE
        )
        
        class_names = train_ds.class_names
        print(f"Bulunan sınıflar (etiketler): {class_names}")

        # 5. Adım 1.5: Veriyi Görselleştirme
        print("Eğitim setinden 9 örnek gösteriliyor...")
        
        plt.figure(figsize=(10, 10))
        for images, labels in train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")
                
        # Grafiği ekranda göster (Bu pencereyi kapatınca kod devam eder)
        plt.show() 
        
    except Exception as e:
        print(f"\n!!!! HATA !!!!")
        print(f"Veri yüklenirken bir hata oluştu: {e}")
        print("Lütfen 'train' klasörünün içinde 'cat' ve 'dog' alt klasörlerinin olduğundan ve içlerinin dolu olduğundan emin olun.")


#---------------------------------------------------
# ADIM 3: MODELİN OLUŞTURULMASI
#---------------------------------------------------
print("Model oluşturuluyor...")

# Temel bir CNN modeli kuruyoruz
model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Modeli derle
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Modelin mimarisini (özetini) terminale yazdır
print("\n--- Model Özeti ---")
model.summary()


#---------------------------------------------------
# ADIM 4: MODELİ EĞİTME
#---------------------------------------------------
EPOCHS = 3 

print(f"\nModel {EPOCHS} epoch (tur) boyunca eğitilmeye başlıyor...")
print("Bu işlem, bilgisayarınızın hızına bağlı olarak uzun sürebilir.")
print("Lütfen sabırla bekleyin...\n")

# model.fit() komutu asıl eğitimi başlatır
history = model.fit(
  train_ds,
  validation_data=validation_ds,
  epochs=EPOCHS
)

print("\nEğitim tamamlandı!")

#---------------------------------------------------
# ADIM 4.5: EĞİTİLMİŞ MODELİ KAYDETME (DOĞRU YER BURASI)
#---------------------------------------------------
model_save_path = 'cats_vs_dogs_model.h5'
print(f"\nModel {model_save_path} olarak kaydediliyor...")

try:
    model.save(model_save_path)
    print("Model başarıyla kaydedildi.")
    print(f"Bu dosyayı daha sonra 'tf.keras.models.load_model(\"{model_save_path}\")' komutuyla yükleyebilirsiniz.")
except Exception as e:
    print(f"Model kaydedilirken bir hata oluştu: {e}")


#---------------------------------------------------
# ADIM 5: EĞİTİM SONUÇLARINI GÖRSELLEŞTİRME
#---------------------------------------------------
print("\nEğitim sonuçları grafiği oluşturuluyor...")

# Düzeltilmiş 'val_accuracy' ve 'val_loss' anahtarları
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Eğitim Doğruluğu (Training Acc)')
plt.plot(epochs_range, val_acc, label='Doğrulama Doğruluğu (Validation Acc)')
plt.legend(loc='lower right')
plt.title('Eğitim ve Doğrulama Başarısı (Accuracy)')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Eğitim Kaybı (Training Loss)')
plt.plot(epochs_range, val_loss, label='Doğrulama Kaybı (Validation Loss)')
plt.legend(loc='upper right')
plt.title('Eğitim ve Doğrulama Kaybı (Loss)')

# Grafikleri göster
plt.show()

print("\nProje tamamlandı!")