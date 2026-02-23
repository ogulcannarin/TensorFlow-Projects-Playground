import tensorflow as tf

a = tf.constant(5)
b = tf.constant(3)
toplam = tf.add(a,b)
print("işlem sonucu:", toplam.numpy())