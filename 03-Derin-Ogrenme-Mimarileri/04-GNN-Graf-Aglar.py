import tensorflow as tf
import numpy as np

# 1. Veri Hazırlama (Basit bir sosyal ağ grafiği düşünelim)
# 4 Kişimiz olsun (Düğümler). Her birinin 2 özelliği var (Örn: Yaş, İlgi Alanı)
features = np.array([
    [1.0, 10.0], # Kişi 0
    [2.0, 20.0], # Kişi 1
    [3.0, 30.0], # Kişi 2
    [4.0, 40.0]  # Kişi 3
], dtype=np.float32)

# Bağlantı Matrisi (Adjacency Matrix): Kim kiminle arkadaş?
# Örneğin: Kişi 0, Kişi 1 ile bağlı.
adj = np.array([
    [0, 1, 0, 0],
    [1, 0, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 0]
], dtype=np.float32)

# 2. Basit bir Graph Convolution (Evrişim) Katmanı Tasarlayalım
def graph_conv_layer(adj, features, weights):
    # Önce komşulardan gelen bilgileri topla (Matris çarpımı ile)
    # Bu adımda her kişi, arkadaş olduğu kişilerin özelliklerini öğrenir.
    neighbors_sum = tf.matmul(adj, features)
    # Sonra bu bilgiyi ağırlıklarla çarpıp aktivasyondan geçir
    return tf.nn.relu(tf.matmul(neighbors_sum, weights))

# 3. Model Parametreleri
weights = tf.Variable(tf.random.normal([2, 2])) # 2 giriş özelliği -> 2 çıkış özelliği

# 4. İşlemi Çalıştıralım
output = graph_conv_layer(adj, features, weights)

print("--- GNN İşlem Sonucu ---")
print("Düğümlerin (Kişilerin) yeni özellikleri (Komşularından öğrendikleriyle):\n")
print(output.numpy())

print("\n[Bilgi]: GNN sayesinde her düğüm artık sadece kendi verisini değil,")
print("bağlı olduğu komşularının verisinden gelen özeti de içeriyor!")