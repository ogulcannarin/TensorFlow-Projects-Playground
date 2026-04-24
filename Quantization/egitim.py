import os
import shutil
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- 1. ADIM: KLASÖRLEME HATASINI DÜZELTME ---
# TensorFlow her sınıf için ayrı klasör bekler (cat/ ve dog/)
path = "dogs-vs-cats/train/train"

print("Klasörler kontrol ediliyor...")
os.makedirs(os.path.join(path, "cat"), exist_ok=True)
os.makedirs(os.path.join(path, "dog"), exist_ok=True)

for file in os.listdir(path):
    if file.endswith(".jpg"):
        if "cat" in file.lower():
            shutil.move(os.path.join(path, file), os.path.join(path, "cat", file))
        elif "dog" in file.lower():
            shutil.move(os.path.join(path, file), os.path.join(path, "dog", file))

# --- 2. ADIM: VERİ HAZIRLAMA ---
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# --- 3. ADIM: MODEL MİMARİSİ ---
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- 4. ADIM: EĞİTİM ---
print("Eğitim başlıyor... Bu işlem bilgisayar hızına göre 5-10 dk sürebilir.")
model.fit(train_generator, epochs=3, validation_data=validation_generator)

# --- 5. ADIM: KAYDETME ---
model.save("kedi_kopek_ham_model.h5")
print("İşlem tamam! 'kedi_kopek_ham_model.h5' dosyası oluşturuldu.")