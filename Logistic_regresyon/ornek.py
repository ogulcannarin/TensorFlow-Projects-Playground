import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# VERİYİ YÜKLE
# -------------------------
df = pd.read_csv("diabetes.csv")

print(df.head())
print(df.info())
print(df.describe())

# -------------------------
# 0 DEĞERLERİ DÜZELT
# -------------------------
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in cols:
    df[col] = df[col].replace(0, df[col].mean())

# -------------------------
# GRAFİKLER
# -------------------------

# Distribution
sns.countplot(x="Outcome", data=df)
plt.title("Diabetes Distribution")
plt.savefig("plots/distribution.png")
plt.close()

# Glucose Boxplot
sns.boxplot(x="Outcome", y="Glucose", data=df)
plt.title("Glucose vs Outcome")
plt.savefig("plots/boxplot.png")
plt.close()

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig("plots/heatmap.png")
plt.close()

# Scatter
sns.scatterplot(x="Age", y="BMI", hue="Outcome", data=df)
plt.title("Age vs BMI")
plt.savefig("plots/scatter.png")
plt.close()

# Feature - Target ayır
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
# Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# Tahmin
y_pred = model.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Accuracy: {accuracy:.2f})")
plt.savefig("plots/confusion_matrix.png")
plt.close()
