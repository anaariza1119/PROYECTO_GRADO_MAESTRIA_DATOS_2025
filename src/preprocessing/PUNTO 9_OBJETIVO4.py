import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib

# =======================
# 1. Cargar datos
# =======================
df = pd.read_csv(
    r"C:\Users\jorge\OneDrive\PROYECTO_GRADO_MAESTRIA_DATOS_2025\data\interim\df_3_4_Merged.csv"
)

# =======================
# 2. Crear variable binaria
# =======================
df["label_resistente"] = df["resistencia_nivel"].apply(lambda x: 1 if x == "ALTO" else 0)

# =======================
# 3. Definir features y target
# =======================
X = df[["genome_id", "genome_name", "family", "species", "genus", "antibiotic"]]
y = df["label_resistente"]

cat_cols = ["genome_name", "family", "species", "genus", "antibiotic"]

# =======================
# 4. Dividir train/test
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =======================
# 5. Pipeline sin SHAP
# =======================
preprocess = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder="drop"
)

model = LogisticRegression(
    max_iter=200,
    solver="saga",
    n_jobs=-1,
    class_weight="balanced"
)

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("logreg", model)
])

# =======================
# 6. Entrenar modelo
# =======================
print("Entrenando Logistic Regression ...")
pipeline.fit(X_train, y_train)

# =======================
# 7. Predicciones
# =======================
y_pred = pipeline.predict(X_test)

# =======================
# 8. Métricas
# =======================
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("F1-score:", f1)
print("Confusion matrix:\n", cm)
print(report)

# =======================
# 9. Guardar métricas en TXT
# =======================
with open("metricas_modelo.txt", "w") as f:
    f.write("Accuracy: " + str(accuracy) + "\n")
    f.write("F1-score: " + str(f1) + "\n\n")
    f.write("Classification Report:\n" + report)

print("Métricas guardadas en metricas_modelo.txt")

# =======================
# 10. Guardar matriz de confusión como imagen
# =======================
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Matriz de confusión")
plt.colorbar()
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.savefig("matriz_confusion.png")
plt.close()

print("Matriz guardada en matriz_confusion.png")

# =======================
# 11. Exportar importancias (coeficientes)
# =======================
encoder = pipeline.named_steps["preprocess"].named_transformers_["categorical"]
feature_names = encoder.get_feature_names_out(cat_cols)

coefs = pipeline.named_steps["logreg"].coef_[0]

df_importances = pd.DataFrame({
    "feature": feature_names,
    "coef": coefs
}).sort_values(by="coef", ascending=False)

df_importances.to_csv("importancias_logit.csv", index=False)

print("Coeficientes guardados en importancias_logit.csv")

# =======================
# 12. Guardar modelo completo
# =======================
joblib.dump(pipeline, "modelo_logit.joblib")

print("Modelo guardado en modelo_logit.joblib")

print("\n--- RESUMEN ---")
print("Accuracy:", accuracy)
print("F1:", f1)
print("Matriz de confusión:\n", cm)



#==============================================================
# ============================================================
# 14. MODELO 3: Naive Bayes (BernoulliNB)
# ============================================================

from sklearn.naive_bayes import BernoulliNB

print("\nEntrenando Bernoulli Naive Bayes ...")

nb_model = BernoulliNB()

pipeline_nb = Pipeline([
    ("preprocess", preprocess),
    ("nb", nb_model)
])

pipeline_nb.fit(X_train, y_train)

y_pred_nb = pipeline_nb.predict(X_test)

accuracy_nb = accuracy_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)
cm_nb = confusion_matrix(y_test, y_pred_nb)
report_nb = classification_report(y_test, y_pred_nb)

print("\n=== RESULTADOS NAIVE BAYES ===")
print("Accuracy:", accuracy_nb)
print("F1-score:", f1_nb)
print("Confusion matrix:\n", cm_nb)
print(report_nb)

with open("metricas_naive_bayes.txt", "w") as f:
    f.write("Accuracy: " + str(accuracy_nb) + "\n")
    f.write("F1-score: " + str(f1_nb) + "\n\n")
    f.write("Classification Report:\n" + report_nb)

plt.figure(figsize=(6,5))
plt.imshow(cm_nb, interpolation="nearest", cmap="Blues")
plt.title("Matriz de confusión – Naive Bayes")
plt.colorbar()
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig("matriz_confusion_naive_bayes.png")
plt.close()

joblib.dump(pipeline_nb, "modelo_naive_bayes.joblib")

print("Modelo Naive Bayes guardado en modelo_naive_bayes.joblib")

# ============================================================
# 16. MODELO 5: XGBoost con SMOTE
# ============================================================

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

print("\nEntrenando XGBoost + SMOTE ...")

# SMOTE solo para entrenamiento
smote = SMOTE(
    sampling_strategy=0.4,   # Ajusta cuánto quieres oversampling de la clase 1
    k_neighbors=3,
    n_jobs=-1
)

# XGBoost optimizado para datos grandes
xgb_smote_model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",     # MUCH faster
    n_jobs=-1,
    random_state=42,
    scale_pos_weight=1      # lo dejamos en 1 porque SMOTE ya balancea
)

# Pipeline con preprocess + SMOTE + XGB
pipeline_xgb_smote = ImbPipeline(steps=[
    ("preprocess", preprocess),   # OneHotEncoder
    ("smote", smote),             # SMOTE solo en el entrenamiento
    ("xgb", xgb_smote_model)
])

# Entrenar modelo
pipeline_xgb_smote.fit(X_train, y_train)

# Predicción
y_pred_xgb_smote = pipeline_xgb_smote.predict(X_test)

# Métricas
accuracy_xgb_smote = accuracy_score(y_test, y_pred_xgb_smote)
f1_xgb_smote = f1_score(y_test, y_pred_xgb_smote)
cm_xgb_smote = confusion_matrix(y_test, y_pred_xgb_smote)
report_xgb_smote = classification_report(y_test, y_pred_xgb_smote)

print("\n=== RESULTADOS XGBOOST + SMOTE ===")
print("Accuracy:", accuracy_xgb_smote)
print("F1-score:", f1_xgb_smote)
print("Confusion matrix:\n", cm_xgb_smote)
print(report_xgb_smote)

# Guardar métricas
with open("metricas_xgboost_smote.txt", "w") as f:
    f.write("Accuracy: " + str(accuracy_xgb_smote) + "\n")
    f.write("F1-score: " + str(f1_xgb_smote) + "\n\n")
    f.write("Classification Report:\n" + report_xgb_smote)

# Matriz de confusión
plt.figure(figsize=(6,5))
plt.imshow(cm_xgb_smote, interpolation="nearest", cmap="Blues")
plt.title("Matriz de confusión – XGBoost + SMOTE")
plt.colorbar()
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig("matriz_confusion_xgb_smote.png")
plt.close()

# Guardar modelo
joblib.dump(pipeline_xgb_smote, "modelo_xgboost_smote.joblib")

print("Modelo XGBoost + SMOTE guardado en modelo_xgboost_smote.joblib")
