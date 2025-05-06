import joblib

features = joblib.load("model_ciktilari/ozellik_isimleri.pkl")
for i, f in enumerate(features, start=1):
    print(f"{i}. {f}")
