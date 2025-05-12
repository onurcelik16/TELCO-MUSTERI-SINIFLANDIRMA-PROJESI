import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from tensorflow.keras.models import load_model


# Model ve bileşen yolları
MODEL_DIR = "model_ciktilari"
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Bileşenleri yükle
model = load_model(f"{MODEL_DIR}/musteri_kaybi_modeli.h5")
scaler = joblib.load(f"{MODEL_DIR}/olceklendirici.pkl")
feature_names = joblib.load(f"{MODEL_DIR}/ozellik_isimleri.pkl")

# Veriyi oku ve işle (proje.py ile aynı şekilde)
df = pd.read_csv(DATA_PATH)
df.dropna(inplace=True)
df = df[df['TotalCharges'] != " "]
df['TotalCharges'] = df['TotalCharges'].astype(float)
df['SözleşmeTipi'] = df['Contract']
df['SözleşmeSüresi'] = df['tenure']
df['HizmetSayısı'] = df[['PhoneService', 'InternetService', 'OnlineSecurity',
                         'PaperlessBilling']].apply(lambda x: sum(x == 'Yes'), axis=1)
df['AylıkGelir'] = df['MonthlyCharges']
df['ToplamGelir'] = df['TotalCharges']
df['HizmetYoğunluğu'] = df['HizmetSayısı'] / (df['tenure'] + 1)

df.drop(columns=['customerID', 'Churn'], inplace=True)
df = pd.get_dummies(df)

# Eksik feature'ları sıfırla
for col in feature_names:
    if col not in df.columns:
        df[col] = 0

# Doğru sırayla al
X = df[feature_names]

# Özellik önem skorları (modelin ilk katmanının ağırlıklarıyla)
weights = model.layers[0].get_weights()[0]  # shape: (input_dim, units)
importance = np.mean(np.abs(weights), axis=1)

# Görselleştir
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(20)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel("Önemsellik Skoru")
plt.title("En Önemli Özellikler (İlk Katmandan)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "feature_importance.png"))

print("✅ Özellik önem grafiği kaydedildi: feature_importance.png")
