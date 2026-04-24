import tensorflow as tf
import numpy as np

# 1. Ham modeli yükle
model = tf.keras.models.load_model("kedi_kopek_ham_model.h5")

# 2. TFLite Dönüştürücüsünü hazırla
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 3. Temsili Veri Seti (Representative Dataset)
# Bu kısım çok önemli: Modele "sayıları yuvarlarken şu örnek verilere bak" diyoruz.
def representative_data_gen():
    # Eğitim verilerinden rastgele 100 örnek alalım (sembolik)
    for _ in range(100):
        data = np.random.rand(1, 128, 128, 3).astype(np.float32)
        yield [data]

converter.representative_dataset = representative_data_gen

# 4. Tam sayı (Integer) nicemlemesini zorunlu tut
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# 5. Dönüştür ve Kaydet
tflite_model_quant = converter.convert()

with open("kedi_kopek_8bit.tflite", "wb") as f:
    f.write(tflite_model_quant)

print("Nicemleme tamamlandı! 'kedi_kopek_8bit.tflite' dosyası hazır.")