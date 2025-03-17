import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Veri setini yükle
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# "Churn" sütununu sayısala çevir
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# TotalCharges sütununu düzelt
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')  # Sayısala çevir
df = df.assign(TotalCharges=df['TotalCharges'].fillna(df['TotalCharges'].median()))

# Kategorik sütunları Label Encoding ile dönüştür
le = LabelEncoder()
categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Bağımsız (özellikler) ve bağımlı (hedef) değişkenleri ayır
X = df.drop(columns=['customerID', 'Churn'])
y = df['Churn']

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hiperparametre optimizasyonu için GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# En iyi modeli seç
best_model = grid_search.best_estimator_
print("En iyi parametreler:", grid_search.best_params_)

# Test seti üzerinde tahmin yap
y_pred = best_model.predict(X_test)

# Doğruluk oranı
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk: {accuracy:.4f}")



# Sınıflandırma raporu
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# Karışıklık matrisi
cmp = confusion_matrix(y_test, y_pred)
sns.heatmap(cmp, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title("Karışıklık Matrisi")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()

# Özellik önem derecesi
importances = best_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names, hue=feature_names, dodge=False, legend=False, palette="viridis")
plt.title("Özellik Önem Dereceleri")
plt.xlabel("Önem Skoru")
plt.ylabel("Özellikler")
plt.show()
