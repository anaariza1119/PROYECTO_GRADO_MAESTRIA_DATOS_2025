
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import re
from difflib import get_close_matches



#################################################### 1. Cargar archivos TSV

df_resistencia = pd.read_csv("C:/Users/jorge/OneDrive/Documentos/BV-BRC/resistencia.tsv", sep="\t")
df_metadata = pd.read_csv("C:/Users/jorge/OneDrive/Documentos/BV-BRC/metadata_genomas_V1.tsv", sep="\t")
#df_features_genomicos = pd.read_csv("C:/Users/jorge/OneDrive/Documentos/BV-BRC/features_genomicos.tsv", sep="\t")
df_multi_sample_table= pd.read_excel("C:/Users/jorge/OneDrive/Documentos/BV-BRC/multi_sample_table2.xlsx", sheet_name='Hoja 4')

 
# Convertir todas las columnas posibles a tipo numérico
df_multi_sample_table = df_multi_sample_table.apply(pd.to_numeric, errors='ignore')


# Aplicar valor absoluto solo a las columnas numéricas
df_multi_sample_table[df_multi_sample_table.select_dtypes(include=[np.number]).columns] = \
    df_multi_sample_table.select_dtypes(include=[np.number]).abs()

#Name_Bacteria
###################################################################################
# Solución corregida: reemplazar nulos con el promedio de cada FILA, únicamente en columnas numéricas


# Identificar columnas numéricas
numeric_cols = df_multi_sample_table.select_dtypes(include='number').columns

# Reemplazar valores nulos con el promedio de cada fila en columnas numéricas
df_multi_sample_table[numeric_cols] = df_multi_sample_table[numeric_cols].T.fillna(df_multi_sample_table[numeric_cols].mean(axis=1)).T

####################################################################################
df_metadata.rename(columns={'genome.genome_id': 'genome_id'}, inplace=True)
df_metadata.rename(columns={'genome.genome_name': 'genome_name'}, inplace=True)
# 1.1. Convertir genome_id a string

df_resistencia['genome_id'] = df_resistencia['genome_id'].astype(str)
df_metadata['genome_id'] = df_metadata['genome_id'].astype(str)

df_multi_sample_table['Name_Bacteria'] = df_multi_sample_table['Name_Bacteria'].astype(str)
#CAMBIO DE NOMBRE PARA CRUCE
df_multi_sample_table.rename(columns={'Name_Bacteria': 'genome_name'}, inplace=True)



df_metadata['genome_name_SP'] =df_metadata['genome_name']



df_merge=df_metadata

# Función para limpiar nombre de bacteria quitando la cepa
def limpiar_nombre(nombre):
    if pd.isnull(nombre):
        return ""
    # Mantener solo las dos primeras palabras (usualmente género y especie)
    partes = re.split(r'\s+', nombre.strip().lower())
    return ' '.join(partes[:2]) if len(partes) >= 2 else partes[0]

# Limpiar columnas
df_multi_sample_table['genome_name'] = df_multi_sample_table['genome_name'].apply(limpiar_nombre)
df_merge['genome_name'] = df_merge['genome_name'].apply(limpiar_nombre)

# Emparejar nombres aproximados
coincidencias = {}
for name in df_multi_sample_table['genome_name'].unique():
    match = get_close_matches(name, df_merge['genome_name'].unique(), n=1, cutoff=0.85)
    if match:
        coincidencias[name] = match[0]

# Agregar columna de coincidencias y hacer merge
df_multi_sample_table['matched_name'] = df_multi_sample_table['genome_name'].map(coincidencias)

df_final = pd.merge(df_multi_sample_table, df_merge, left_on='matched_name', right_on='genome_name', how='inner')


df_validar= df_final['genome_name_x'].unique()



# 1.3. Estandarizar nombres de columnas
df_final.columns = df_final.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('genome.', '').str.replace('genome_drug.', '')

# 1.4. Filtrar solo bacterias con huésped humano
df_final = df_final[df_final['host_name'].str.lower().str.contains('homo sapiens|human', na=False)]

# 1.5. Eliminar columnas duplicadas
#df_final = df_final.loc[:, ~df_final.columns.duplicated()]

# 1.6. Eliminar filas completamente vacías y duplicadas
#df_final.dropna(how='all', inplace=True)
#df_final.drop_duplicates(inplace=True)

# 1.7. Limpiar espacios y caracteres especiales en columnas de texto
for col in df_final.select_dtypes(include=['object']):
    df_final[col] = df_final[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)

# 1.8. Convertir columnas numéricas
if 'collection_year' in df_final.columns:
    df_final['collection_year'] = pd.to_numeric(df_final['collection_year'], errors='coerce')

# 1.9. Rellenar valores nulos en campos clave
for col in ['host_name', 'isolation_country', 'resistant_phenotype']:
    if col in df_final.columns:
        df_final[col] = df_final[col].fillna('unknown')
        
# 1.10. Normalizar valores de resistant_phenotype
if 'resistant_phenotype' in df_final.columns:
    df_final['resistant_phenotype'] = df_final['resistant_phenotype'].str.lower().replace({
        'resistant': 'resistant',
        'susceptible': 'susceptible',
        'intermediate': 'intermediate',
        'nan': 'unknown'
    })


df_final = df_final.drop('genome_name_y', axis=1)
df_final = df_final.drop('genome_name_x', axis=1)
df_final = df_final.drop('host_name', axis=1)

#df_final.to_excel("C:/Users/jorge/OneDrive/Documentos/BV-BRC/metadata_genomas_sin_host.xlsx", index=False)
#df_final.head(1000).to_excel("C:/Users/jorge/OneDrive/Documentos/BV-BRC/metadata_genomas_sin_host.xlsx", index=False)

import numpy as np



# Filtramos las columnas por tipo de paciente
placa_cols = [col for col in df_final.columns if col.startswith('placa_')]
saliva_cols = [col for col in df_final.columns if col.startswith('saliva_')]
tumor_cols = [col for col in df_final.columns if col.startswith('turmor_')]

# Concatenamos los datos con etiquetas
df_placa = df_final[placa_cols].T.dropna(axis=1).copy()
df_placa['grupo'] = 'placa'

df_saliva = df_final[saliva_cols].T.dropna(axis=1).copy()
df_saliva['grupo'] = 'saliva'

df_tumor = df_final[tumor_cols].T.dropna(axis=1).copy()
df_tumor['grupo'] = 'tumor'

# Unimos todos los grupos
df_union = pd.concat([df_placa, df_saliva, df_tumor], axis=0)

# Separar características y etiquetas
X = df_union.drop(columns='grupo').values
y = df_union['grupo'].values

# Aplicamos PCA para reducir a 2 dimensiones
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Crear un DataFrame para graficar
df_plot = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_plot['grupo'] = y

# Graficar los datos en el plano cartesiano
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_plot, x='PC1', y='PC2', hue='grupo', s=100, alpha=0.8)
plt.title('Segmentación de pacientes por grupo')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.grid(True)
plt.legend(title='Grupo')
plt.tight_layout()
plt.show()


################################################# 2. Análisis exploratorio
# a. Antibióticos más frecuentes
top_antibioticos = df_final['antibiotic'].value_counts().head(10)
print("Antibióticos más frecuentes:")
print(top_antibioticos)

# b. Distribución por país
top_paises = df_final['isolation_country'].value_counts().head(10)
print("\nPaíses con más aislamientos:")
print(top_paises)

# c. Tendencia por año de recolección
df_final['collection_year'] = pd.to_numeric(df_final['collection_year'], errors='coerce')
conteo_por_ano = df_final['collection_year'].value_counts().sort_index()

# Gráfico de barras: antibióticos
plt.figure(figsize=(10, 6))
sns.countplot(y='antibiotic', data=df_final, order=df_final['antibiotic'].value_counts().iloc[:10].index)
plt.title("Antibióticos más comunes")
plt.xlabel("Frecuencia")
plt.ylabel("Antibiótico")
plt.tight_layout()
plt.show()

# Gráfico de barras: país
plt.figure(figsize=(10, 6))
sns.countplot(y='isolation_country', data=df_final, order=df_final['isolation_country'].value_counts().iloc[:10].index)
plt.title("Países con más registros")
plt.xlabel("Frecuencia")
plt.ylabel("País")
plt.tight_layout()
plt.show()

# Línea de tiempo: registros por año
plt.figure(figsize=(10, 5))
conteo_por_ano.plot(kind='line', marker='o')
plt.title("Registros de genomas por año")
plt.xlabel("Año de recolección")
plt.ylabel("Cantidad de genomas")
plt.grid(True)
plt.tight_layout()
plt.show()


# Top 10 antibióticos más frecuentes
top_abx = df_final['antibiotic'].value_counts().head(10).index
abx_df = df_final[df_final['antibiotic'].isin(top_abx)]

# Agrupar y obtener el tipo de resistencia más común por antibiótico
resumen_abx = abx_df.groupby('antibiotic')['resistant_phenotype'].agg(lambda x: x.value_counts().idxmax()).reset_index()
resumen_abx.columns = ['Antibiótico', 'Fenotipo más frecuente']

# Cantidad total por antibiótico
frecuencias = df_final['antibiotic'].value_counts().head(10).reset_index()
frecuencias.columns = ['Antibiótico', 'Frecuencia']

# Unir ambas tablas
resumen_final = pd.merge(frecuencias, resumen_abx, on='Antibiótico')
#######################################3. MODELOS 




# Filtrar por humanos y limpiar
df_final = df_final[df_final['host_name'].str.lower().str.contains("homo sapiens")]
df_final = df_final.drop_duplicates()
df_final = df_final.dropna(subset=['resistant_phenotype', 'antibiotic', 'genome_name'])

# Codificación
le = LabelEncoder()
df_final['antibiotic'] = le.fit_transform(df_final['antibiotic'].astype(str))
df_final['genome_name'] = le.fit_transform(df_final['genome_name'].astype(str))
df_final['isolation_country'] = le.fit_transform(df_final['isolation_country'].astype(str))
df_final['resistant_phenotype'] = le.fit_transform(df_final['resistant_phenotype'].astype(str))
df_final['collection_year'] = df_final['collection_year'].fillna(df_final['collection_year'].mode()[0])

# Variables
X = df_final[['antibiotic', 'genome_name', 'isolation_country', 'collection_year']]
y = df_final['resistant_phenotype']

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------------------
# Modelo 1: Regresión Logística
# ------------------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\n Regresión Logística")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# ------------------------------
# Modelo 2: Random Forest
# ------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n Random Forest")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ------------------------------
# Modelo 3: XGBoost
# ------------------------------
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("\n XGBoost")
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# ------------------------------
# Importancia de variables
# ------------------------------
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', title="Importancia de Variables - Random Forest")
plt.tight_layout()
plt.show()
#



##---------------------------------------------------------------------------------------------
#¿Qué significa?
#El modelo no alcanzó la convergencia dentro del número máximo de iteraciones (max_iter=1000). Esto no invalida el modelo, pero indica que los coeficientes no se ajustaron completamente.
# Soluciones posibles:
#Escalar los datos numéricos con StandardScaler
#Aumentar el número de iteraciones: LogisticRegression(max_iter=5000)
#Usar otro solver: solver="saga" o solver="liblinear" si es clasificación binaria
# Resultados del modelo
#Regresión Logística
#Precisión general (accuracy): 0.51 → modelo débil
#Problema: ¡no predijo ninguna muestra para las clases 0 y 3!
#Conclusión: Este modelo no es adecuado en este contexto con datos multiclase y desbalanceados.
#Random Forest
#Precisión general (accuracy): 0.71 → rendimiento aceptable
#Clases 1 y 2 se predicen bastante bien
#Clase 3: F1-score = 0.50 → podría mejorarse
#Clase 0: rendimiento bajo, probablemente porque hay pocos ejemplos (solo 20)
# Conclusión: Este modelo es el más estable de los tres para predicción general.
#XGBoost
#Precisión general (accuracy): 0.74 → el mejor modelo en términos generales
#Clase 3: buen desempeño (F1-score = 0.57)
#Clase 0: mejora frente a RF (F1 = 0.39), pero aún limitada por pocos ejemplos
# Conclusión: XGBoost es el mejor clasificador multiclase para tu caso. Puedes usar este resultado como base para tus conclusiones de tesis.
# Recomendaciones para mejorar más:
#Balanceo de clases: aplica SMOTE o class_weight="balanced" para mejorar clases minoritarias (0 y 3).
#Feature Engineering: incluye nuevas variables del entorno bacteriano si las tienes.
#Evaluación visual: agrega gráficas de matrices de confusión y curvas ROC si presentas esto.


# Filtrar humanos
df_final = df_final[df_final['host_name'].str.lower().str.contains("homo sapiens")]
df_final = df_final.drop_duplicates()
df_final = df_final.dropna(subset=['resistant_phenotype', 'antibiotic', 'genome_name'])

# Codificación
le = LabelEncoder()
df_final['antibiotic'] = le.fit_transform(df_final['antibiotic'].astype(str))
df_final['genome_name'] = le.fit_transform(df_final['genome_name'].astype(str))
df_final['isolation_country'] = le.fit_transform(df_final['isolation_country'].astype(str))
df_final['resistant_phenotype'] = le.fit_transform(df_final['resistant_phenotype'].astype(str))
df_final['collection_year'] = df_final['collection_year'].fillna(df_final['collection_year'].mode()[0])

# Variables
X = df_final[['antibiotic', 'genome_name', 'isolation_country', 'collection_year']]
y = df_final['resistant_phenotype']

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Función para mostrar matriz de confusión
def mostrar_matriz(modelo, y_true, y_pred, nombre):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Matriz de Confusión - {nombre}")
    plt.show()

# ------------------------------
# Modelo 1: Regresión Logística
# ------------------------------
lr = LogisticRegression(max_iter=5000, class_weight='balanced', solver='lbfgs')
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\n Regresión Logística")
print(classification_report(y_test, y_pred_lr))
mostrar_matriz(lr, y_test, y_pred_lr, "Regresión Logística")

# ------------------------------
# Modelo 2: Random Forest
# ------------------------------
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n Random Forest")
print(classification_report(y_test, y_pred_rf))
mostrar_matriz(rf, y_test, y_pred_rf, "Random Forest")

# ------------------------------
# Modelo 3: XGBoost
# ------------------------------
xgb = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("\n XGBoost")
print(classification_report(y_test, y_pred_xgb))
mostrar_matriz(xgb, y_test, y_pred_xgb, "XGBoost")

# Importancia de variables
importances = pd.Series(rf.feature_importances_, index=['antibiotic', 'genome_name', 'isolation_country', 'collection_year'])
importances.sort_values().plot(kind='barh', title="Importancia de Variables - Random Forest")
plt.tight_layout()
plt.show()


#"Se probaron tres modelos de aprendizaje supervisado: Regresión Logística, Random Forest y XGBoost. La Regresión Logística presentó un bajo desempeño general (accuracy del 22%), evidenciando limitaciones para la clasificación multiclase en este contexto. Random Forest y XGBoost mostraron desempeños robustos, con precisiones superiores al 70%. El modelo XGBoost obtuvo el mejor desempeño global (F1 ponderado = 0.74), siendo seleccionado como modelo final para caracterizar patrones de resistencia antibiótica en bacterias asociadas al ser humano."

#####################################MODELOS de CLUSTERING
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Reducir dimensiones para visualización
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Clustering con KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Visualización
plt.figure(figsize=(8,5))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette='Set2')
plt.title("Clustering de bacterias (PCA + KMeans)")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.show()

#########PASO 2 CLUSTERING
df_final['cluster'] = clusters  # clusters viene del modelo KMeans

# Ejemplo: distribución de antibióticos por clúster
for i in sorted(df_final['cluster'].unique()):
    print(f"\n Cluster {i} - Total muestras: {len(df_final[df_final['cluster'] == i])}")
    print(df_final[df_final['cluster'] == i]['antibiotic'].value_counts().head(5))


plt.figure(figsize=(10,5))
sns.countplot(data=df_final, x='resistant_phenotype', hue='cluster')
plt.title("Distribución de resistencia por clúster")
plt.show()
#CLUSTERING
#Se encontraron 4 agrupamientos naturales entre las bacterias orales muestreadas.
#Cada grupo se distingue por combinaciones específicas de antibióticos, fenotipo de resistencia y origen epidemiológico.
#El análisis puede servir como base para vigilancia epidemiológica o predicción temprana de resistencia.


# Asignar X y y Un árbol de decisión que predice el tipo de resistencia según las variables seleccionadas.
X = df_final.drop(columns=['cluster'])
y = df_final['cluster']

# Convertir categóricas
X_encoded = pd.get_dummies(X, drop_first=True)

# Entrenar modelo
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_encoded, y)

# Visualizar
plt.figure(figsize=(20,10))
plot_tree(tree, feature_names=X_encoded.columns, class_names=[str(i) for i in tree.classes_], filled=True)
plt.show()


# Asumiendo que tienes el cluster en df_final['cluster']
resumen = df_final.groupby('cluster')[['genome_name', 'antibiotic', 'host_name', 'isolation_country']].agg(lambda x: x.mode().iloc[0])
print(resumen)


# Generar un perfil de cada clúster bacteriano

for c in df_final['cluster'].unique():
    print(f"\n Perfil del cluster {c}")
    print(df_final[df_final['cluster'] == c]['resistant_phenotype'].value_counts(normalize=True))
    print(df_final[df_final['cluster'] == c]['genome_name'].value_counts().head(3))

#Visualización de perfiles por clúster (heatmap)
pivot = pd.crosstab(df_final['cluster'], df_final['resistant_phenotype'])
sns.heatmap(pivot, annot=True, cmap='coolwarm')
plt.title('Distribución de resistencia por clúster')
plt.ylabel('Cluster')
plt.xlabel('Fenotipo de resistencia')
plt.show()

#"Ciertos clústeres bacterianos tienen mayor prevalencia de resistencia múltiple"


# Evaluar si hay diferencia significativa entre los clústeres en cuanto a tipo de resistencia
from scipy.stats import chi2_contingency

tabla = pd.crosstab(df_final['cluster'], df_final['resistant_phenotype'])
chi2, p, dof, expected = chi2_contingency(tabla)

print(f"p-value = {p}")
if p < 0.05:
    print(" Hay diferencia significativa en la resistencia entre clústeres.")
else:
    print(" No hay diferencia significativa.")
    
##significa que existe una relación estadísticamente significativa entre los clústeres bacterianos identificados y el fenotipo de resistencia. Es decir, los grupos de bacterias formados por clustering presentan patrones de resistencia distintos.

#¿Qué implica esto para tu tesis?
#Valida tu hipótesis central: los patrones de resistencia a antibióticos están fuertemente relacionados con grupos específicos de bacterias.
#Justifica el uso de clustering y modelos de clasificación como una herramienta efectiva para identificar riesgos clínicos o guiar decisiones en salud pública.
#Puedes proponer el uso de este enfoque como una herramienta de vigilancia epidemiológica o para priorizar tratamientos según los perfiles bacterianos.

