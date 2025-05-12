import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Veri ve model çıktı dizini
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = "model_ciktilari"
os.makedirs(MODEL_DIR, exist_ok=True)

# ✅ Veri hazırlama fonksiyonu
def hazirla_veri():
    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)
    df = df[df['TotalCharges'] != " "]
    df['TotalCharges'] = df['TotalCharges'].astype(float)

    # Yeni sütunlar
    df['SözleşmeTipi'] = df['Contract']
    df['SözleşmeSüresi'] = df['tenure']
    df['HizmetSayısı'] = df[['PhoneService', 'InternetService', 'OnlineSecurity', 'PaperlessBilling']].apply(lambda x: sum(x == 'Yes'), axis=1)
    df['AylıkGelir'] = df['MonthlyCharges']
    df['ToplamGelir'] = df['TotalCharges']
    df['HizmetYoğunluğu'] = df['HizmetSayısı'] / (df['tenure'] + 1)

    df.drop(columns=['customerID'], inplace=True)
    y = df['Churn'].replace({'Yes': 1, 'No': 0})
    df.drop(columns=['Churn'], inplace=True)

    # Kategorik verileri sayısallaştır
    df = pd.get_dummies(df)
    X = df.drop(columns=['SözleşmeTipi'], errors='ignore')

    # Ölçekleme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Özellik isimlerini kaydet
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "ozellik_isimleri.pkl"))

    # SMOTE ile veri dengeleme
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    return X_resampled, y_resampled, scaler

# ✅ Model oluşturma ve eğitme
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
                        validation_data=(X_val, y_val), callbacks=[early_stop], verbose=1)

    # Eğitim grafikleri
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Eğitim Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Grafiği')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Eğitim Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Grafiği')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "egitim_loss_accuracy.png"))
    plt.clf()

    return model, history

# ✅ Modeli değerlendirme
def degerlendir(model, X_test, y_test):
    y_pred_prob = model.predict(X_test).ravel()
    best_threshold = 0.50
    y_pred = (y_pred_prob >= best_threshold).astype(int)

    # Sınıflandırma raporu
    report = classification_report(y_test, y_pred)
    with open(os.path.join(MODEL_DIR, "siniflandirma_raporu.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
    plt.clf()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.savefig(os.path.join(MODEL_DIR, "precision_recall_curve.png"))
    plt.clf()

    # ROC Curve ve AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(os.path.join(MODEL_DIR, "roc_curve.png"))
    plt.clf()

    return best_threshold

# ✅ Model ve yardımcı bileşenleri kaydetme
def kaydet(model, scaler, threshold):
    model.save(os.path.join(MODEL_DIR, "musteri_kaybi_modeli.h5"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "olceklendirici.pkl"))
    joblib.dump(threshold, os.path.join(MODEL_DIR, "optimal_esik.pkl"))
    print("Model ve bileşenler başarıyla kaydedildi.")

# ✅ Ana çalışma bloğu
if __name__ == "__main__":
    X, y, scaler = hazirla_veri()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model, history = egit_model(X_train, y_train, X_test, y_test)
    threshold = degerlendir(model, X_test, y_test)
    kaydet(model, scaler, threshold)
