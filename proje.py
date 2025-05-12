import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = "model_ciktilari"

os.makedirs(MODEL_DIR, exist_ok=True)

def hazirla_veri():
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

    df.drop(columns=['customerID'], inplace=True)

    y = df['Churn'].replace({'Yes': 1, 'No': 0})
    df.drop(columns=['Churn'], inplace=True)

    df = pd.get_dummies(df)
    X = df.drop(columns=['SözleşmeTipi'], errors='ignore')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "ozellik_isimleri.pkl"))

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    return X_resampled, y_resampled, scaler

def egit_model(X_train, y_train, X_val, y_val):
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stop], verbose=1)
    return model, history

def degerlendir(model, X_test, y_test):
    y_pred_prob = model.predict(X_test).ravel()

    best_threshold = 0.50  # Sabit eşik değeri

    y_pred = (y_pred_prob >= best_threshold).astype(int)
    report = classification_report(y_test, y_pred)

    with open(os.path.join(MODEL_DIR, "siniflandirma_raporu.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
    plt.clf()

    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.savefig(os.path.join(MODEL_DIR, "precision_recall_curve.png"))
    plt.clf()

    print("Sabit eşik değeriyle değerlendirme tamamlandı. Eşik:", best_threshold)
    return best_threshold

def kaydet(model, scaler, threshold):
    model.save(os.path.join(MODEL_DIR, "musteri_kaybi_modeli.h5"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "olceklendirici.pkl"))
    joblib.dump(threshold, os.path.join(MODEL_DIR, "optimal_esik.pkl"))
    print("Model ve bileşenler başarıyla kaydedildi.")

if __name__ == "__main__":
    X, y, scaler = hazirla_veri()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model, history = egit_model(X_train, y_train, X_test, y_test)
    threshold = degerlendir(model, X_test, y_test)
    kaydet(model, scaler, threshold)
