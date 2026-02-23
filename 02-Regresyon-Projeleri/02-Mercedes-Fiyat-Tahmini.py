import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

# VERİYİ OKU
# Veri seti datasets klasörüne taşındığı için yol güncellendi.
df = pd.read_excel("../datasets/merc.xlsx")

print(df.head())
print(df.describe())
print(df.isnull().sum())

# PRICE DAĞILIMI
plt.figure(figsize=(7,5))
sbn.histplot(df["price"], kde=True)
plt.show()

# YEAR DAĞILIMI
plt.figure(figsize=(7,5))
sbn.countplot(x="year", data=df)
plt.show()

# MILEAGE - PRICE İLİŞKİSİ
plt.figure(figsize=(7,5))
sbn.scatterplot(x="mileage", y="price", data=df)
plt.show()

# EN PAHALI VE EN UCUZ 20 ARAÇ
print(df.sort_values("price", ascending=False).head(20))
print(df.sort_values("price", ascending=True).head(20))

print("Toplam veri:", len(df))
print("Yüzde 1:", len(df)*0.01)

# EN PAHALI %1'İ AT
yuzdedoksandokuzdf = df.sort_values("price", ascending=False).iloc[131:]

plt.figure(figsize=(7,5))
sbn.histplot(yuzdedoksandokuzdf["price"], kde=True)
plt.show()

# SADECE PRICE ÜZERİNDEN ORTALAMA
print(df.groupby("year")["price"].mean())
print(yuzdedoksandokuzdf.groupby("year")["price"].mean())
print(df[df.year != 1970].groupby("year")["price"].mean())

# YENİ DF
df = yuzdedoksandokuzdf

print(df.describe())

# 1970 MODELİ SİL
df = df[df.year != 1970]

print(df.groupby("year")["price"].mean())
print(df.head())

# TRANSMISSION SİL (String kolon problemi olmasın)
df = df.drop("transmission", axis=1)

# X ve Y AYIR
y = df["price"].values
x = df.drop("price", axis=1).values

print(x[:5])
print(y[:5])

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=10
)

print(len(x_train))
print(len(x_test))

# SCALE
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime
import os

model = Sequential()
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

# CALLBACKS
# EarlyStopping: Val loss iyileşmeyi bıraktığında eğitimi durdurur.
early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=15)

# TensorBoard: Eğitim sürecini görselleştirmek için log tutar.
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size=128,
    epochs=150,
    callbacks=[early_stopping, tensorboard_callback]
)

# MODELİ KAYDET
if not os.path.exists("models"):
    os.makedirs("models")
model.save("models/mercedes_model.h5")
print("Model 'models/mercedes_model.h5' olarak kaydedildi.")

# LOSS GRAFİĞİ
loss_df = pd.DataFrame(history.history)

plt.figure(figsize=(7,5))
loss_df.plot()
plt.title("Eğitim ve Doğrulama Kaybı")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()

# TAHMİN VE ANALİZ
tahminDizisi = model.predict(x_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("MAE:", mean_absolute_error(y_test, tahminDizisi))
print("MSE:", mean_squared_error(y_test, tahminDizisi))

plt.figure(figsize=(7,5))
plt.scatter(y_test, tahminDizisi)
plt.plot(y_test, y_test, "g-*") # Tam doğru çizgisi
plt.title("Gerçek vs Tahmin Edilen Değerler")
plt.xlabel("Gerçek Fiyat")
plt.ylabel("Tahmin Edilen Fiyat")
plt.show()
