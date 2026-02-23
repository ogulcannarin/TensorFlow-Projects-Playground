import tensorflow as tf
import numpy as np

X =np.array([1,2,3,4,5,6,] ,dtype=np.float32)
Y = np.array([10,20,30,40,50,60],dtype=np.float32)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(optimizer="sgd" , loss="mean_squared_error")
print("Eğitim başlıyor...")
model.fit(X,Y, epochs=500 , verbose=0)
print("Eğitim tamamlandı.")

print("Tahmin Sonunu  ) : ",model.predict([10.0]))
