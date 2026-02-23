import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------------------------------------
# 1) Veriyi Okuma
# -------------------------------------------------
# Veri seti datasets klasörüne taşındığı için yol güncellendi.
df = pd.read_excel("../datasets/bisiklet_fiyatlari.xlsx")
df.columns = df.columns.str.strip()

print(df.head())
print(df.columns)

# -------------------------------------------------
# 2) Feature - Label Ayırma
# -------------------------------------------------
y = df["Fiyat"].values
x = df[["BisikletOzellik1", "BisikletOzellik2"]].values

# -------------------------------------------------
# 3) Train - Test Bölme
# -------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=15
)

# -------------------------------------------------
# 4) Scaling
# -------------------------------------------------
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# -------------------------------------------------
# 5) Model Oluşturma
# -------------------------------------------------
model = Sequential([
    Dense(5, activation="relu"),
    Dense(5, activation="relu"),
    Dense(5, activation="relu"),
    Dense(1)
])

model.compile(optimizer="rmsprop", loss="mse")

# -------------------------------------------------
# 6) Model Eğitme
# -------------------------------------------------
history = model.fit(
    x_train,
    y_train,
    epochs=250,
    validation_data=(x_test, y_test),
    verbose=1
)

# -------------------------------------------------
# 7) Test Loss
# -------------------------------------------------
test_loss = model.evaluate(x_test, y_test)
print("Test Loss:", test_loss)

# -------------------------------------------------
# 8) Loss Grafiği
# -------------------------------------------------
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# -------------------------------------------------
# 9) Tahmin ve Gerçek Karşılaştırma
# -------------------------------------------------
test_predictions = model.predict(x_test).flatten()

tahminDf = pd.DataFrame({
    "Gerçek Y": y_test,
    "Tahmin Y": test_predictions
})

plt.figure()
sbn.scatterplot(x="Gerçek Y", y="Tahmin Y", data=tahminDf)

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.title("Gerçek vs Tahmin")
plt.show()
