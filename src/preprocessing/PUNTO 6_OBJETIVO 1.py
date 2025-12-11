# -*- coding: utf-8 -*-
import os
import re
import unicodedata
import pandas as pd
import numpy as np
import chardet
import matplotlib.pyplot as plt


# Configuración general
KEEP_FULL_IN_MEMORY_2_5 = False   # <- Agrega esta línea
CHUNK_SIZE = 200_000

# ============================================================
# 6	OBTENCIÓN Y PREPARACIÓN DE LOS DATOS
# 6.3 PREPARACIÓN Y LIMPIEZA DE DATOS
# - Normalización de columnas
# - Depuración de vacíos / duplicados
# - Estandarización taxonómica (Género + especie)
# ============================================================

# ---------- Rutas ----------
ruta_externa = r"C:\Users\jorge\OneDrive\PROYECTO_GRADO_MAESTRIA_DATOS_2025\data\external"
ruta_raw = r"C:\Users\jorge\OneDrive\PROYECTO_GRADO_MAESTRIA_DATOS_2025\data\raw"
ruta_out = r"C:\Users\jorge\OneDrive\PROYECTO_GRADO_MAESTRIA_DATOS_2025\data\interim"
os.makedirs(ruta_out, exist_ok=True)

path_001 = os.path.join(ruta_externa, "001_DATOS_MUESTRAS_BACTERIANAS.xlsx")
path_002 = os.path.join(ruta_raw,     "002_DATOS_ID_BACTERIANOS.tsv")
path_003 = os.path.join(ruta_raw,     "003_DATOS_FAMILIAS.tsv")
path_004 = os.path.join(ruta_raw,     "004_DATOS_RESISTENCIA.tsv")

# ---------- UTILIDADES ----------


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(" ", "_")
          .str.replace("[^a-z0-9_]", "", regex=True)
    )
    return df


def limpiar_textos_y_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza general:
      - Strings: strip/lower; vacíos comunes -> NaN.
      - Elimina filas totalmente vacías y duplicadas.S
      - Numéricas: NaN -> 0; negativos -> 0.
      - Texto: NaN -> "0" y, si el texto representa un número negativo, -> "0".
    """
    df = df.copy()
    vacios = {"", " ", "-", "na", "n/a", "NA", "N/A"}

    # --- 1) Limpieza de texto base ---
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(
                lambda x: x.strip().lower() if isinstance(x, str) else x)
            df[col] = df[col].apply(lambda x: np.nan if isinstance(
                x, str) and x in vacios else x)

    # --- 2) Quitar filas vacías y duplicadas ---
    df = df.dropna(how="all").drop_duplicates()

    # --- 3) Numéricas reales: NaN -> 0; negativos -> 0 ---
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        df[num_cols] = df[num_cols].fillna(0)
        df[num_cols] = df[num_cols].applymap(lambda x: 0 if x < 0 else x)

    # --- 4) Objetos que contienen números negativos en texto ---
    # patrón: acepta guion normal o unicode (−) y decimales con punto o coma
    neg_pattern = re.compile(r"^\s*[−-]\s*\d+(?:[.,]\d+)?\s*$")

    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        # a) reemplazar NaN por "0"
        df[col] = df[col].fillna("0")

        # b) detectar strings que representan números negativos
        mask_neg_text = df[col].astype(str).apply(
            lambda s: bool(neg_pattern.match(s)))

        # c) para esas filas, poner "0"
        if mask_neg_text.any():
            df.loc[mask_neg_text, col] = "0"

        # d) adicional: intentar coercer a número para detectar negativos ocultos tipo " -3,5 "
        tmp = (
            df[col]
            .astype(str)
            .str.replace("−", "-", regex=False)   # guion unicode -> normal
            .str.replace(",", ".", regex=False)   # coma decimal -> punto
        )
        tmp_num = pd.to_numeric(tmp, errors="coerce")
        mask_neg_num = tmp_num < 0
        if mask_neg_num.any():
            df.loc[mask_neg_num, col] = "0"

    return df


def limpiar_nombre_cientifico(nombre: str) -> str | None:
    """
    Estandariza a 'Género especie':
      - Quita tildes, corchetes, paréntesis, comillas y puntos.
      - Elimina 'sp', 'spp' y metadatos de cepa (strain, ATCC, DSM, etc.).
      - NO fragmenta tokens con '_' o '-' (se descartan completos si no son alfabéticos).
      - Evita tomar como 'especie' términos de sitio/aislado comunes (eye, skin, soil, etc.).
    """
    import re
    import unicodedata

    if not isinstance(nombre, str) or not nombre.strip():
        return None

    # 1) Normalización básica
    n = ''.join(c for c in unicodedata.normalize(
        'NFD', nombre) if unicodedata.category(c) != 'Mn')
    n = re.sub(r'[\[\]\(\)\'\"{}\.]', '', n)

    # 2) Cortar metadatos de cepa/colecciones y lo que siga
    n = re.sub(r'\b(strain|type|isolate|atcc|dsm|nctc|ccug|jcm|kctc|nbrc|cbs|subsp|subspecies|umb|mmrc|cw|we)\b.*',
               '', n, flags=re.IGNORECASE)

    # 3) Quitar sp/spp como palabras completas
    n = re.sub(r'\bsp\.?\b|\bspp\.?\b', '', n, flags=re.IGNORECASE)

    # 4) Espacios prolijos (¡sin reemplazar '_' ni '-'!)
    n = re.sub(r'\s+', ' ', n).strip()

    # 5) Tokenizar por ESPACIO y quedarse solo con tokens 100% alfabéticos
    tokens_orig = n.split()
    tokens_alpha = [t for t in tokens_orig if re.fullmatch(r'[A-Za-z]+', t)]

    if not tokens_alpha:
        return None

    genero = tokens_alpha[0].capitalize()

    # Lista corta de términos no-especie comunes (puedes ampliarla si ves otros)
    STOP_ESPECIE = {
        "eye", "skin", "soil", "water", "aquifer", "sediment", "lake", "river", "seawater", "marine",
        "stool", "feces", "fecal", "urine", "blood", "saliva", "oral", "nasal", "vaginal", "gut",
        "sample", "isolate", "host", "human", "mouse", "rat", "pig", "cow", "bovine", "canine", "feline"
    }

    if len(tokens_alpha) >= 2:
        especie_candidata = tokens_alpha[1].lower()
        if especie_candidata not in STOP_ESPECIE:
            return f"{genero} {especie_candidata}"

    # Si no hay epíteto válido (o cae en stoplist), devuelve solo el género
    return genero


def write_backup_csv(df: pd.DataFrame, path_out: str):
    df.to_csv(path_out, index=False, encoding="utf-8")
    print("Backup →", path_out, df.shape)


def normalizar_tsv_en_chunks(path_in: str, out_name: str,
                             crear_tax: bool = False, col_tax_src: str = None,
                             keep_full_in_memory: bool = False):

    out_path = os.path.join(ruta_out, out_name)
    if os.path.exists(out_path):
        os.remove(out_path)

    sample_frames = []
    header_written = False
    total_in = total_out = dups_removed = 0

    # ----- lector robusto por encoding -----
    try:
        reader = pd.read_csv(
            path_in,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
            low_memory=True,
            encoding="utf-8"       # intento principal
        )
    except UnicodeDecodeError:
        print(f" UTF-8 falló en {os.path.basename(path_in)
                                 }. Reintentando con ISO-8859-1…")
        reader = pd.read_csv(
            path_in,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
            low_memory=True,
            encoding="ISO-8859-1"  # fallback clásico
        )

    # ----- proceso por bloques -----
    for ch in reader:
        ch = normalizar_columnas(ch)

        rows_before = len(ch)
        dups_before = ch.duplicated().sum()

        # Limpieza (incluye pasar negativos a 0 dentro de tu función)
        ch = limpiar_textos_y_nulos(ch).drop_duplicates()
        rows_after = len(ch)
        dups_after = ch.duplicated().sum()

        total_in += rows_before
        total_out += rows_after
        dups_removed += (dups_before - dups_after)

        # Estandarización taxonómica si se solicita
        if crear_tax and col_tax_src and col_tax_src in ch.columns:
            ch["nombre_normalizado"] = ch[col_tax_src].apply(
                limpiar_nombre_cientifico)

        # Escribir backup incremental
        ch.to_csv(out_path, mode="a", header=not header_written,
                  index=False, encoding="utf-8")
        header_written = True

        # Mantener muestra o todo en memoria
        sample_frames.append(ch if keep_full_in_memory else ch.head(500))

    df_mem = pd.concat(
        sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    print(f"Backup → {out_path} | filas in={total_in} → out={
          total_out} | dups removidos={dups_removed}")
    return df_mem


# ---------- 001 (Excel) ----------
df_normalizado_001 = pd.read_excel(path_001)
df_normalizado_001 = normalizar_columnas(df_normalizado_001)
antes = (len(df_normalizado_001), df_normalizado_001.duplicated().sum())

df_normalizado_001 = limpiar_textos_y_nulos(df_normalizado_001)
df_normalizado_001 = df_normalizado_001.drop_duplicates()

despues = (len(df_normalizado_001), df_normalizado_001.duplicated().sum())
print(f"001: filas {antes[0]} → {
      despues[0]} | dups removidos: {antes[1]-despues[1]}")

if "name_bacteria" in df_normalizado_001.columns:
    df_normalizado_001["nombre_normalizado"] = df_normalizado_001["name_bacteria"].apply(
        limpiar_nombre_cientifico)

write_backup_csv(df_normalizado_001, os.path.join(
    ruta_out, "001_normalizado.csv"))

# ---------- 002 (TSV) con nombre_normalizado desde genome.genome_name ----------
df_normalizado_002 = normalizar_tsv_en_chunks(
    path_002,
    out_name="002_normalizado.csv",
    crear_tax=True,
    col_tax_src="genomegenome_name",
    keep_full_in_memory=KEEP_FULL_IN_MEMORY_2_5
)

# ---------- 003–005 (TSV) ----------
df_normalizado_003 = normalizar_tsv_en_chunks(path_003, "003_normalizado.csv",
                                              keep_full_in_memory=KEEP_FULL_IN_MEMORY_2_5)
df_normalizado_004 = normalizar_tsv_en_chunks(path_004, "004_normalizado.csv",
                                              keep_full_in_memory=KEEP_FULL_IN_MEMORY_2_5)

# ---------- VISTAZOS RÁPIDOS ----------
print("\nHead 001:")
print(df_normalizado_001.head(3))
print("\nHead 002:")
print(df_normalizado_002.head(3))
print("\nHead 003:")
print(df_normalizado_003.head(3))
print("\nHead 004:")
print(df_normalizado_004.head(3))

# ---------------------------------------------
# Df que se relaciona con el 002
# ---------------------------------------------
df001_unicos = (
    df_normalizado_001[["nombre_normalizado"]]
    .drop_duplicates()
    .reset_index(drop=True)
)  # uso para relacionar

# Trae las rutas en donde se almacenan los archivos ya normalizados
rn001 = os.path.join(ruta_out, "001_normalizado.csv")  # uso para AED
rn002 = os.path.join(ruta_out, "002_normalizado.csv")
rn003 = os.path.join(ruta_out, "003_normalizado.csv")
rn004 = os.path.join(ruta_out, "004_normalizado.csv")

# Crea los DF
df001 = pd.read_csv(rn001, dtype=str, low_memory=False)
df002 = pd.read_csv(rn002, dtype=str, low_memory=False)
df003 = pd.read_csv(rn003, dtype=str, low_memory=False)
df004 = pd.read_csv(rn004, dtype=str, low_memory=False)

# Elimina duplicados
df001 = df001.drop_duplicates()
df002 = df002.drop_duplicates()
df003 = df003.drop_duplicates()
df004 = df004.drop_duplicates()


# ----------------------------------------------
# 6.4 INTEGRACION DE DATOS
# ----------------------------------------------

# Paleta de colores unificada
PALETTE = ['#2E86C1', '#F39C12', '#27AE60',
           '#C0392B', '#8E44AD', '#16A085', '#7F8C8D']

# ----------------------------------------------
# Paso 1 - Asegurar tipo str
# ----------------------------------------------


def _ensure_str(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

# ----------------------------------------------
# Paso 2 - Pie de coincidencias
# ----------------------------------------------


def pie_match_vs_nomatch(matched_count, base_count, title, out_path=None):
    matched = int(matched_count)
    base = int(base_count) if base_count else 0
    no_match = max(base - matched, 0)
    pct_m = round((matched / base * 100), 2) if base else 0
    pct_nm = round((no_match / base * 100), 2) if base else 0

    sizes = [matched, no_match]
    labels = ['Cruzan', 'No cruzan']
    colors = [PALETTE[0], PALETTE[1]]
    explode = (0.05, 0)

    plt.figure(figsize=(6, 6))
    wedges, texts, autotexts = plt.pie(
        sizes,
        explode=explode,
        labels=None,
        colors=colors,
        startangle=90,
        shadow=True,
        autopct='%1.1f%%',
        pctdistance=0.75,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.legend(
        wedges,
        [f"{lab} ({val/base*100:.1f}%)" for lab,
         val in zip(labels, sizes) if base > 0],
        loc='upper left',
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=10
    )
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return {
        "matched": matched,
        "nomatch": no_match,
        "base": base,
        "pct_matched": pct_m,
        "pct_nomatch": pct_nm
    }

# ----------------------------------------------
# Paso 3 - Unión y reportes
# ----------------------------------------------


def export_join_reports(left_df, right_df, key, left_name, right_name, out_dir, base_is='left'):
    os.makedirs(out_dir, exist_ok=True)

    inner = left_df.merge(right_df, how='inner', on=key,
                          suffixes=(f"_{left_name}", f"_{right_name}"))
    left_only = left_df[~left_df[key].isin(inner[key])]
    right_only = right_df[~right_df[key].isin(inner[key])]

    if base_is == 'left':
        base_total = left_df[key].nunique(dropna=True)
    elif base_is == 'inner':
        base_total = inner[key].nunique(dropna=True)
    else:
        base_total = left_df[key].nunique(dropna=True)

    matched_total = inner[key].nunique(dropna=True)

    inner_path = os.path.join(
        out_dir, f"inner_{left_name}_vs_{right_name}.csv")
    left_only_path = os.path.join(
        out_dir, f"no_{right_name}_en_{left_name}.csv")
    right_only_path = os.path.join(
        out_dir, f"no_{left_name}_en_{right_name}.csv")

    inner.to_csv(inner_path, index=False, encoding='utf-8')
    left_only.to_csv(left_only_path, index=False, encoding='utf-8')
    right_only.to_csv(right_only_path, index=False, encoding='utf-8')

    pie_path = os.path.join(out_dir, f"pie_{left_name}_vs_{right_name}.png")
    title = f"Relación entre {left_name} y {right_name}"
    pie_stats = pie_match_vs_nomatch(
        matched_total, base_total, title, pie_path)

    resumen_row = {
        "union": f"{left_name}_vs_{right_name}",
        "clave": key,
        "base": base_is,
        "base_total": base_total,
        "matched": pie_stats["matched"],
        "no_matched": pie_stats["nomatch"],
        "pct_matched": pie_stats["pct_matched"],
        "pct_no_matched": pie_stats["pct_nomatch"],
        "inner_csv": inner_path,
        "left_only_csv": left_only_path,
        "right_only_csv": right_only_path,
        "pie_png": pie_path
    }
    return inner, left_only, right_only, resumen_row

# ----------------------------------------------
# Paso 4 - Pie analíticos
# ----------------------------------------------


def pie_from_series(series, title, out_path=None, top_n=None, drop_values=None):
    s = series.astype(str).str.strip()
    if drop_values:
        s = s[~s.isin(drop_values)]

    vc = s.value_counts()
    if top_n is not None and len(vc) > top_n:
        top = vc.head(top_n)
        otros = vc.iloc[top_n:].sum()
        vc = pd.concat([top, pd.Series({'Otros': otros})])

    sizes = vc.values
    labels = vc.index
    colors = PALETTE[:len(labels)]
    explode = (0.05,) + (0,) * (len(labels) - 1)

    plt.figure(figsize=(6, 6))
    wedges, texts, autotexts = plt.pie(
        sizes,
        explode=explode,
        labels=None,
        colors=colors,
        startangle=90,
        shadow=True,
        autopct='%1.1f%%',
        pctdistance=0.75,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('equal')
    total = sum(sizes)
    plt.legend(
        wedges,
        [f"{lab} ({val/total*100:.1f}%)" for lab,
         val in zip(labels, sizes) if total > 0],
        loc='upper left',
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=10
    )
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ----------------------------------------------
# Paso 5 - Integraciones
# ----------------------------------------------
df001_unicos = _ensure_str(df001_unicos, ['nombre_normalizado'])
df002 = _ensure_str(df002, ['nombre_normalizado'])

inner_001_002, no_002_en_001, no_001_en_002, row_001_002 = export_join_reports(
    left_df=df001_unicos,
    right_df=df002,
    key='nombre_normalizado',
    left_name='001',
    right_name='002',
    out_dir=os.path.join(ruta_out, "001_vs_002"),
    base_is='left'
)

df001_002 = inner_001_002.rename(columns={
    "genomegenome_id": "genome_id",
    "genomegenome_name": "genome_name"
})
df001_002 = _ensure_str(df001_002, ['genome_id'])
df003 = _ensure_str(df003, ['genome_id'])

inner_001002_003, no_003_en_001002, no_001002_en_003, row_001002_003 = export_join_reports(
    left_df=df001_002[['nombre_normalizado',
                       'genome_id', 'genome_name']].drop_duplicates(),
    right_df=df003[['genome_id']].drop_duplicates(),
    key='genome_id',
    left_name='001_002',
    right_name='003',
    out_dir=os.path.join(ruta_out, "001_002_vs_003"),
    base_is='left'
)

df004 = _ensure_str(df004, ['genome_id'])
inner_0010023_004, no_004_en_0010023, no_0010023_en_004, row_0010023_004 = export_join_reports(
    left_df=inner_001002_003[['genome_id']].drop_duplicates(),
    right_df=df004[['genome_id']].drop_duplicates(),
    key='genome_id',
    left_name='001_002_003',
    right_name='004',
    out_dir=os.path.join(ruta_out, "001_002_003_vs_004"),
    base_is='left'
)

# ----------------------------------------------
# Paso 6 - Resumen general
# ----------------------------------------------
resumen = pd.DataFrame([row_001_002, row_001002_003, row_0010023_004])
resumen_path = os.path.join(ruta_out, "resumen_integracion.csv")
resumen.to_csv(resumen_path, index=False, encoding='utf-8')
print("Resumen guardado en:", resumen_path)

# ----------------------------------------------
# Paso 7 - Pies analíticos
# ----------------------------------------------
df_004_inner_full = inner_001002_003.merge(
    df004, how='inner', on='genome_id', suffixes=("", "_004")
)
valores_invalidos = {"0", "0.0", "00", "nan", "NaN", "", "None", "NULL"}

pie_from_series(
    series=df_004_inner_full['evidence'],
    title='Distribución de Evidence',
    out_path=os.path.join(ruta_out, "pies_analiticos", "pie_evidence.png"),
    top_n=12,
    drop_values=valores_invalidos
)

pie_from_series(
    series=df_004_inner_full['resistant_phenotype'],
    title='Distribución de Resistant Phenotype',
    out_path=os.path.join(ruta_out, "pies_analiticos",
                          "pie_resistant_phenotype.png"),
    top_n=12
)

pie_from_series(
    series=df_004_inner_full['resistant_phenotype'],
    title='Distribución de Resistant Phenotype (sin ceros)',
    out_path=os.path.join(ruta_out, "pies_analiticos",
                          "pie_resistant_phenotype_sin_ceros.png"),
    top_n=12,
    drop_values=valores_invalidos
)

# ----------------------------------------------
# Paso 8 - Mostrar resumen
# ----------------------------------------------
print("\n========== RESUMEN DE INTEGRACIÓN ==========\n")
for fila in [row_001_002, row_001002_003, row_0010023_004]:
    print(f" Unión: {fila['union']}")
    print(f"  - Base de referencia: {fila['base']}")
    print(f"  - Total base: {fila['base_total']:,}")
    print(f"  - Coinciden: {fila['matched']:,} ({fila['pct_matched']}%)")
    print(f"  - No coinciden: {fila['no_matched']
          :,} ({fila['pct_no_matched']}%)")
    print(f"  - CSV coincidencias: {os.path.basename(fila['inner_csv'])}")
    print(
        f"  - CSV no coinciden (der): {os.path.basename(fila['right_only_csv'])}")
    print(f"  - Gráfico: {os.path.basename(fila['pie_png'])}")
    print("-" * 60)

df_004_inner_full.to_csv(
    "C:\\Users\\jorge\\OneDrive\\PROYECTO_GRADO_MAESTRIA_DATOS_2025\\data\\interim\\df_004_inner_full.csv", index=False, encoding='utf-8')

#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------



# ---------------------------------Limpieza para OBJETIVO 3----------------------------------------
# ---------- 1) RUTAS: ajusta según donde estén tus CSV ----------
PATH = '.'  # carpeta actual; cámbialo si los CSV están en otra carpeta
file_fenotipos = os.path.join(
    PATH, r"C:\Users\jorge\OneDrive\PROYECTO_GRADO_MAESTRIA_DATOS_2025\data\interim\003_normalizado.csv")
file_resistencias = os.path.join(
    PATH, r"C:\Users\jorge\OneDrive\PROYECTO_GRADO_MAESTRIA_DATOS_2025\data\interim\004_normalizado.csv")

# ---------- 2) CARGA ----------
df_feno = pd.read_csv(file_fenotipos, dtype="string",
                      engine="python").astype(str)
df_res = pd.read_csv(file_resistencias, dtype="string",
                     engine="python").astype(str)


print("Fenotipos:", df_feno.shape)
print("Resistencias:", df_res.shape)


# Normalizar texto (importante para evitar diferencias por mayúsculas/minúsculas)
df_res['resistant_phenotype'] = (
    df_res['resistant_phenotype']
    .str.strip().str.lower()
)

# Diccionario de recodificación → 3 niveles
map_resistencia = {
    'resistant': 'ALTO',
    'nonsusceptible': 'ALTO',
    'reduced susceptibility': 'MEDIO',
    'intermediate': 'MEDIO',
    'susceptible-dose dependent': 'MEDIO',
    'susceptible': 'BAJO',
    '0': 'BAJO',
    '0.0': 'BAJO',
    '': 'BAJO'
}

# Aplicar recodificación
df_res['resistencia_nivel'] = df_res['resistant_phenotype'].map(
    map_resistencia)
print(df_res['resistencia_nivel'].value_counts())

# --------------3) MERGUE ----------
df = df_res.merge(df_feno[['genome_id', 'family', 'genus', 'species']],
                  on='genome_id', how='left')

print("Merged:", df.shape)
print(df[['family', 'genus', 'species', 'resistencia_nivel']].head())

# Lista de columnas que deseas conservar
columnas_requeridas = [
    "genome_id",
    "genome_name",
    "family",
    "species",
    "genus",
    "antibiotic",
    "resistencia_nivel"
]

# Filtrar el dataframe
df_filtrado = df[columnas_requeridas]


# Mostrar dimensiones
print("Tamaño del nuevo DF:", df_filtrado.shape)




df_filtrado.to_csv("C:\\Users\\jorge\\OneDrive\\PROYECTO_GRADO_MAESTRIA_DATOS_2025\\data\\interim\\df_3_4_Merged.csv",
          index=False, encoding='utf-8')
