# eda_relacional_resistencia.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import chi2_contingency

# Crear colormap basado en el azul principal de tus barplots
blue_custom = LinearSegmentedColormap.from_list(
    "blue_custom",
    ["#0B3C5D", "#2E86C1", "#A9CCE3"]   # oscuro → azul barras → claro
)
PALETTE = ['#2E86C1', '#F39C12', '#27AE60', '#C0392B', '#8E44AD', '#16A085', '#7F8C8D']

sns.set_theme(style="whitegrid")
sns.set_palette(PALETTE)

df = pd.read_csv(
    r"C:\Users\jorge\OneDrive\PROYECTO_GRADO_MAESTRIA_DATOS_2025\data\interim\df_3_4_Merged.csv"
)

# -------------------------------------------------------------------------------------------
# FAMILY VS RESISTANCE
# -------------------------------------------------------------------------------------------

ct_family = pd.crosstab(df['family'], df['resistencia_nivel'])
chi2, p, dof, expected = chi2_contingency(ct_family)
print("FAMILY vs resistencia_nivel")
print("Chi2:", chi2, " p-value:", p, " df:", dof)

plt.figure(figsize=(10,8))
sns.heatmap(
    ct_family,
    cmap=blue_custom,  # <--- TU NUEVO COLORMAP
    linewidths=0.4,
    linecolor="white"
)
plt.title("Distribución de resistencia_nivel por FAMILY")
plt.ylabel("Family")
plt.xlabel("Resistant phenotype")
plt.tight_layout()
plt.show()

ct_family_prop = ct_family.div(ct_family.sum(axis=1), axis=0)
ct_family_prop.plot(kind="bar", stacked=True, figsize=(12,6), color=PALETTE)
plt.title("Proporción de fenotipos de resistencia por familia")
plt.ylabel("Proporción")
plt.tight_layout()
plt.show()


# -------------------------------------------------------------------------------------------
# GENUS VS RESISTANCE (Top 20)
# -------------------------------------------------------------------------------------------

top_genus = df['genus'].value_counts().head(20).index
df_top = df[df['genus'].isin(top_genus)]

ct_genus = pd.crosstab(df_top['genus'], df_top['resistencia_nivel'])
chi2, p, dof, expected = chi2_contingency(ct_genus)

print("GENUS vs resistant_phenotype")
print("Chi2:", chi2, " p-value:", p, " df:", dof)

plt.figure(figsize=(12,7))
sns.heatmap(
    ct_genus,
    cmap=blue_custom,  # <--- UNIFICADO
    linewidths=0.4,
    linecolor="white"
)
plt.title("Heatmap de resistencia por Género (Top 20)")
plt.tight_layout()
plt.show()

ct_genus_prop = ct_genus.div(ct_genus.sum(axis=1), axis=0)
ct_genus_prop.plot(kind="bar", stacked=True, figsize=(14,6), color=PALETTE)
plt.title("Proporciones de fenotipos resistentes por Género (Top 20)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# -------------------------------------------------------------------------------------------
# SPECIES VS RESISTANCE (Top 20)
# -------------------------------------------------------------------------------------------

top_species = df['species'].value_counts().head(20).index
df_s = df[df['species'].isin(top_species)]

ct_species = pd.crosstab(df_s['species'], df_s['resistencia_nivel'])
chi2, p, dof, expected = chi2_contingency(ct_species)

print("SPECIES vs resistant_phenotype")
print("Chi2:", chi2, " p-value:", p, " df:", dof)

plt.figure(figsize=(14,8))
sns.heatmap(
    ct_species,
    cmap=blue_custom,  # <--- UNIFICADO
    linewidths=0.4,
    linecolor="white"
)
plt.title("Heatmap resistencia por Especie (Top 20)")
plt.tight_layout()
plt.show()

ct_species_prop = ct_species.div(ct_species.sum(axis=1), axis=0)
ct_species_prop.plot(kind="bar", stacked=True, figsize=(14,6), color=PALETTE)
plt.title("Proporción de resistencia por Especie (Top 20)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# -------------------------------------------------------------------------------------------
# PROPORCIONES
# -------------------------------------------------------------------------------------------

prop_family = df.groupby(["family", "resistencia_nivel"]).size().unstack(fill_value=0)
print("\n=== PROPORCIONES DE RESISTENCIA POR FAMILY ===")
print(prop_family)

prop_genus = df.groupby(["genus", "resistencia_nivel"]).size().unstack(fill_value=0)
print("\n=== PROPORCIONES DE RESISTENCIA POR GENUS ===")
print(prop_genus)

prop_species = df.groupby(["species", "resistencia_nivel"]).size().unstack(fill_value=0)
print("\n=== PROPORCIONES DE RESISTENCIA POR ESPECIES ===")
print(prop_species)
