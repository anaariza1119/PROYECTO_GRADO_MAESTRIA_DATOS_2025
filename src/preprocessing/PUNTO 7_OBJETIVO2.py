# =========================================================
# 7. ANÁLISIS EXPLORATORIO DE MUESTRAS CLÍNICAS
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ---------------------------------------------------------
# Paleta de colores unificada (misma del punto 6)
# ---------------------------------------------------------
PALETTE = ['#2E86C1', '#F39C12', '#27AE60', '#C0392B', '#8E44AD', '#16A085', '#7F8C8D']

sns.set_theme(style="whitegrid")
sns.set_palette(PALETTE)

# ---------------------------------------------------------
# Paso 1: Cargar datos
# ---------------------------------------------------------
ruta = "C:\\Users\\jorge\\OneDrive\\PROYECTO_GRADO_MAESTRIA_DATOS_2025\\data\\interim\\001_normalizado.csv"  # Ajustar según tu ubicación real
df = pd.read_csv(ruta)

print("Dimensiones:", df.shape)
print("Columnas principales:", df.columns[:10].tolist())

# ---------------------------------------------------------
# Paso 2: Identificar grupos de muestras
# ---------------------------------------------------------
grupos = {
    'CP': [c for c in df.columns if 'control' in c and 'placa' in c],
    'MP': [c for c in df.columns if 'placa_paciente' in c],
    'CS': [c for c in df.columns if 'saliva_control' in c],
    'MS': [c for c in df.columns if 'saliva_paciente' in c],
    'MT': [c for c in df.columns if 'turmor_paciente' in c]
}

for g, cols in grupos.items():
    print(f"{g}: {len(cols)} columnas detectadas")

# ---------------------------------------------------------
# Paso 3: Resumen de abundancia por grupo
# ---------------------------------------------------------
resumen = {}
for grupo, cols in grupos.items():
    resumen[grupo] = df[cols].sum(axis=1)

df_resumen = pd.DataFrame(resumen)
df_resumen["nombre_normalizado"] = df["nombre_normalizado"]

# Normalizar proporciones
df_resumen_prop = df_resumen.set_index("nombre_normalizado")
df_resumen_prop = df_resumen_prop.div(df_resumen_prop.sum(axis=0), axis=1)

# ---------------------------------------------------------
# Paso 4: Heatmap – abundancia relativa
# ---------------------------------------------------------
plt.figure(figsize=(10, 12))
sns.heatmap(df_resumen_prop.head(30), cmap="YlGnBu", cbar_kws={'label': 'Proporción relativa'})
plt.title("Abundancia relativa de las 30 bacterias principales", fontsize=14, fontweight='bold')
plt.xlabel("Tipo de muestra", fontsize=12)
plt.ylabel("Bacterias", fontsize=12)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Paso 5: Top bacterias por tipo de muestra (barras multicolor sin error bars)
# ---------------------------------------------------------
top_n = 10
for grupo in grupos.keys():
    top = df_resumen.nlargest(top_n, grupo)[["nombre_normalizado", grupo]]

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=top,
        x=grupo,
        y="nombre_normalizado",
        palette=PALETTE[:len(top)],  # cada barra con color distinto
        errorbar=None  # <- evita mostrar la línea negra
    )

    plt.title(f"Top {top_n} bacterias más abundantes en {grupo}", fontsize=13, fontweight='bold')
    plt.xlabel("Nivel de abundancia", fontsize=11)
    plt.ylabel("Bacteria", fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# Paso 6: Comparación con tumores (MT)
# ---------------------------------------------------------
top_mt = set(df_resumen.nlargest(15, 'MT')["nombre_normalizado"])
otros_totales = df_resumen.drop(columns=['nombre_normalizado', 'MT']).sum(axis=1)
top_otros = set(df.loc[otros_totales.nlargest(15).index, "nombre_normalizado"])

comunes = top_mt.intersection(top_otros)
solo_mt = top_mt.difference(top_otros)

print("\n=== Bacterias presentes tanto en tumores como en otros grupos ===")
for b in comunes:
    print(f" - {b}")
print("\n=== Bacterias exclusivas o dominantes en tumores (MT) ===")
for b in solo_mt:
    print(f" - {b}")

# ---------------------------------------------------------
# Paso 7: Boxplot comparativo por tipo de muestra
# ---------------------------------------------------------
df_box = df_resumen.melt(id_vars="nombre_normalizado", var_name="Tipo_muestra", value_name="Abundancia")

plt.figure(figsize=(8, 6))
sns.boxplot(data=df_box, x="Tipo_muestra", y="Abundancia", palette=PALETTE)
plt.title("Distribución general de abundancias por tipo de muestra", fontsize=14, fontweight='bold')
plt.xlabel("Tipo de muestra", fontsize=12)
plt.ylabel("Abundancia total por bacteria", fontsize=12)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Paso 8: Guardar resumen
# ---------------------------------------------------------
out_path = "resumen_exploratorio_001.csv"
df_resumen.to_csv(out_path, index=False, encoding='utf-8')
print(f"\nResumen de abundancias guardado en: {out_path}")


# =========================================================
# 8. VALIDACIÓN DE ESTRUCTURA – df_004_inner_full.csv
# =========================================================

# =========================================================
# 9. ANÁLISIS EXPLORATORIO TAXONÓMICO Y DE RESISTENCIA
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paleta unificada
PALETTE = ['#2E86C1', '#F39C12', '#27AE60', '#C0392B', '#8E44AD', '#16A085', '#7F8C8D']
sns.set_theme(style="whitegrid")
sns.set_palette(PALETTE)

# Cargar archivo
ruta = "C:\\Users\\jorge\\OneDrive\\PROYECTO_GRADO_MAESTRIA_DATOS_2025\\data\\interim\\df_004_inner_full.csv"
df = pd.read_csv(ruta)

os.makedirs("graficos_resistencia", exist_ok=True)

# ---------------------------------------------------------
# Paso 1: Distribución por nivel taxonómico
# ---------------------------------------------------------
niveles = ['family', 'genus', 'species']
for n in niveles:
    if n in df.columns:
        conteo = df[n].value_counts().head(10)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=conteo.values, y=conteo.index, palette=PALETTE[:len(conteo)], errorbar=None)
        plt.title(f"Top 10 {n.title()} más representativos", fontsize=13, fontweight='bold')
        plt.xlabel("Frecuencia")
        plt.ylabel(n.title())
        plt.tight_layout()
        plt.savefig(f"graficos_resistencia/top10_{n}.png", dpi=300)
        plt.show()

# ---------------------------------------------------------
# Paso 2: Antibióticos más frecuentes
# ---------------------------------------------------------
if 'antibiotic' in df.columns:
    top_antib = df['antibiotic'].value_counts().head(10)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_antib.values, y=top_antib.index, palette=PALETTE[:len(top_antib)], errorbar=None)
    plt.title("Top 10 antibióticos más frecuentes", fontsize=13, fontweight='bold')
    plt.xlabel("Número de ocurrencias")
    plt.ylabel("Antibiótico")
    plt.tight_layout()
    plt.savefig("graficos_resistencia/top10_antibioticos.png", dpi=300)
    plt.show()

# ---------------------------------------------------------
# Paso 3: Distribución de fenotipos de resistencia
# ---------------------------------------------------------
if 'resistant_phenotype' in df.columns:
    fenotipos = df['resistant_phenotype'].value_counts(normalize=True) * 100
    plt.figure(figsize=(6, 6))
    wedges, texts, autotexts = plt.pie(
        fenotipos.values,
        labels=None,
        colors=PALETTE[:len(fenotipos)],
        autopct='%1.1f%%',
        startangle=90,
        shadow=True,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
    )
    plt.legend(wedges, [f"{lab} ({val:.1f}%)" for lab, val in zip(fenotipos.index, fenotipos.values)],
               loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10, frameon=True)
    plt.title("Distribución de fenotipos de resistencia", fontsize=13, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("graficos_resistencia/pie_fenotipos.png", dpi=300)
    plt.show()

# ---------------------------------------------------------
# Paso 4: Relación familia bacteriana - resistencia
# ---------------------------------------------------------
if all(col in df.columns for col in ['family', 'resistant_phenotype']):
    cross = df.groupby(['family', 'resistant_phenotype']).size().reset_index(name='conteo')
    pivot = cross.pivot(index='family', columns='resistant_phenotype', values='conteo').fillna(0)
    pivot = pivot.div(pivot.sum(axis=1), axis=0)
    top_familias = pivot.sum(axis=1).sort_values(ascending=False).head(10).index
    pivot_top = pivot.loc[top_familias]

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_top, cmap="YlOrRd", annot=True, fmt=".2f", cbar_kws={'label': 'Proporción'})
    plt.title("Distribución de fenotipos de resistencia por familia bacteriana", fontsize=13, fontweight='bold')
    plt.xlabel("Fenotipo")
    plt.ylabel("Familia")
    plt.tight_layout()
    plt.savefig("graficos_resistencia/heatmap_familia_resistencia.png", dpi=300)
    plt.show()

# ---------------------------------------------------------
# Paso 5: Distribución de evidencias (laboratorio vs. computador)
# ---------------------------------------------------------
if 'evidence' in df.columns:
    evidencias = df['evidence'].value_counts(normalize=True) * 100
    plt.figure(figsize=(6, 6))
    wedges, texts, autotexts = plt.pie(
        evidencias.values,
        labels=None,
        colors=PALETTE[:len(evidencias)],
        autopct='%1.1f%%',
        startangle=90,
        shadow=True,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
    )
    plt.legend(wedges, [f"{lab} ({val:.1f}%)" for lab, val in zip(evidencias.index, evidencias.values)],
               loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10, frameon=True)
    plt.title("Distribución de tipos de evidencia (laboratorio vs. predicción computacional)",
              fontsize=13, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("graficos_resistencia/pie_evidencias.png", dpi=300)
    plt.show()


