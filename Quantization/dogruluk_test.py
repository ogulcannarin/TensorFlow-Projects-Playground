import tensorflow as tf
import numpy as np

# 1. TFLite modelini yükle
interpreter = tf.lite.Interpreter(model_path="kedi_kopek_8bit.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. Test verisinden bir örnekle deneme yapalım (veya validation_generator kullanabilirsin)
# Şimdilik modelin çalışıp çalışmadığını ve çıktı aralığını görelim
print("Giriş Tipi:", input_details[0]['dtype']) 
# Burada <class 'numpy.uint8'> görmelisin, bu 8-bit olduğunu kanıtlar.

print("Çıkış Tipi:", output_details[0]['dtype'])