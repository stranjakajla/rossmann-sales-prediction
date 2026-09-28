"""
=====================================================================
SEMINARSKI RAD - MASINSKO UCENJE
Predikcija dnevne prodaje po prodavnici (Rossmann Store Sales)
Tip problema: REGRESIJA (target = Sales, kontinuirana varijabla)
Algoritmi: Linearna regresija, Random Forest Regressor
=====================================================================

Struktura:
  1. Ucitavanje i pregled podataka
  2. Spajanje train + store
  3. Feature engineering iz datuma + izbacivanje zatvorenih dana
  4. Provjera i obrada nedostajucih vrijednosti
  5. Deskriptivna statistika (histogrami, boxplotovi)
  6. Analiza kategorickih varijabli
  7. Distribucija ciljne varijable (Sales)
  8. Analiza odnosa izmedju varijabli i ciljne varijable
  9. Enkodiranje kategorickih varijabli
 10. Korelacijska analiza
 11. Preprocessing (skaliranje, definisanje X/y, redukovani skup)
 12. Modeliranje - 6 iteracija
 13. Analiza rezultata (Actual vs Predicted, usporedba, vaznost varijabli)

Svi grafici i CSV s rezultatima cuvaju se u folder ./outputs
=====================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------------
# PODESAVANJA
# ---------------------------------------------------------------
DATA_DIR = "data"       # folder sa train.csv i store.csv (Kaggle)
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
# None = koriste se svi podaci. Ako Random Forest predugo traje,
# postaviti npr. 200000 za treniranje na reprezentativnom uzorku
# (dataset ima 1M+ redova).
SAMPLE_SIZE = None

sns.set_style("whitegrid")
COLOR_LR = "#4C72B0"   # plava - linearna regresija 
COLOR_RF = "#55A868"   # zelena - random forest 


# =================================================================
# 1. UCITAVANJE I PREGLED PODATAKA
# =================================================================
print("=" * 70)
print("1. UCITAVANJE I PREGLED PODATAKA")
print("=" * 70)

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), low_memory=False)
store = pd.read_csv(os.path.join(DATA_DIR, "store.csv"))

print(f"train.csv: {train.shape[0]} redova, {train.shape[1]} kolona")
print(f"store.csv: {store.shape[0]} redova, {store.shape[1]} kolona")
print("\nPrvih 5 redova train.csv:")
print(train.head())
print("\nTipovi podataka (train):")
print(train.dtypes)


# =================================================================
# 2. SPAJANJE TRAIN + STORE
# =================================================================
print("\n" + "=" * 70)
print("2. SPAJANJE PODATAKA (train + store, po koloni Store)")
print("=" * 70)

df = train.merge(store, on="Store", how="left")
print(f"Nakon spajanja: {df.shape[0]} redova, {df.shape[1]} kolona")


# =================================================================
# 3. FEATURE ENGINEERING IZ DATUMA + IZBACIVANJE ZATVORENIH DANA
# =================================================================
print("\n" + "=" * 70)
print("3. FEATURE ENGINEERING IZ DATUMA")
print("=" * 70)

df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
print("Dodane kolone: Year, Month, Day, WeekOfYear")

closed_count = (df["Open"] == 0).sum()
print(f"\nRedova gdje je Open == 0 (prodavnica zatvorena): {closed_count} "
      f"({closed_count / len(df) * 100:.2f}%)")
print("Izbaceno jer je Sales uvijek 0 kad je prodavnica zatvorena - ne nosi "
      "prediktivnu informaciju i vjestacki bi iskrivilo statistiku/model.")
df = df[df["Open"] == 1].copy()
df = df.drop(columns=["Open"])
print(f"Nakon izbacivanja zatvorenih dana: {df.shape[0]} redova")


# =================================================================
# 4. PROVJERA I OBRADA NEDOSTAJUCIH VRIJEDNOSTI
# =================================================================
print("\n" + "=" * 70)
print("4. PROVJERA I OBRADA NEDOSTAJUCIH VRIJEDNOSTI")
print("=" * 70)

missing = df.isna().sum()
missing = missing[missing > 0]
print("Kolone s nedostajucim vrijednostima (prije obrade):")
print(missing)

# --- Grupna imputacija CompetitionDistance po StoreType (napredniji
# pristup nego globalni medijan)
before_impute = df["CompetitionDistance"].copy()
if df["CompetitionDistance"].isna().any():
    n_missing = df["CompetitionDistance"].isna().sum()
    group_medians = df.groupby("StoreType")["CompetitionDistance"].transform("median")
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(group_medians)
    # ako neki StoreType uopste nema poznatu vrijednost, fallback na globalni medijan
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
        df["CompetitionDistance"].median())
    print(f"\nCompetitionDistance: {n_missing} vrijednosti zamijenjeno GRUPNIM "
          f"medijanom po StoreType (umjesto globalnog medijana - preciznije, "
          f"jer razliciti tipovi prodavnica imaju razlicitu tipicnu udaljenost "
          f"konkurencije).")
    print(df.groupby("StoreType")["CompetitionDistance"].median())

# CompetitionOpenSinceMonth/Year - NaN znaci da podatak o konkurenciji
# nije poznat. Popunjavamo sa 0 (placeholder).
for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]:
    n_missing = df[col].isna().sum()
    df[col] = df[col].fillna(0)
    print(f"{col}: {n_missing} NaN vrijednosti zamijenjeno sa 0 "
          f"(oznacava da podatak o otvaranju konkurencije ne postoji)")

# Promo2Since... i PromoInterval - NaN znaci da prodavnica nikad nije
# ucestvovala u Promo2 programu (logicki povezano sa Promo2 == 0)
for col in ["Promo2SinceWeek", "Promo2SinceYear"]:
    n_missing = df[col].isna().sum()
    df[col] = df[col].fillna(0)
    print(f"{col}: {n_missing} NaN vrijednosti zamijenjeno sa 0 "
          f"(prodavnica nikad nije bila u Promo2 programu)")

n_missing = df["PromoInterval"].isna().sum()
df["PromoInterval"] = df["PromoInterval"].fillna("None")
print(f"PromoInterval: {n_missing} NaN vrijednosti zamijenjeno sa 'None'")

print(f"\nProvjera nakon obrade - preostalo NaN vrijednosti: "
      f"{df.isna().sum().sum()}")

# Graf: grupna imputacija prije/poslije
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
axes[0].hist(before_impute.dropna(), bins=40, color="#8172B2", alpha=0.8,
             label="Poznate vrijednosti")
axes[0].set_title("Prije imputacije (NaN izostavljen)")
axes[0].set_xlabel("CompetitionDistance (m)")
axes[0].set_ylabel("Frekvencija")
axes[0].legend()

colors_bar = sns.color_palette("Set2", df["StoreType"].nunique())
group_meds = df.groupby("StoreType")["CompetitionDistance"].median().sort_index()
axes[1].bar(group_meds.index.astype(str), group_meds.values, color=colors_bar)
for i, v in enumerate(group_meds.values):
    axes[1].text(i, v, f"{v:.0f}", ha="center", va="bottom")
axes[1].set_title("Nakon grupne imputacije - medijan po StoreType")
axes[1].set_xlabel("StoreType")
axes[1].set_ylabel("Medijan CompetitionDistance (m)")

plt.suptitle("Grupna imputacija - CompetitionDistance po tipu prodavnice",
              fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_grupna_imputacija.png"), dpi=120)
plt.close()
print("\nGraf sacuvan: outputs/01_grupna_imputacija.png")


# =================================================================
# 5. DESKRIPTIVNA STATISTIKA
# =================================================================
print("\n" + "=" * 70)
print("5. DESKRIPTIVNA STATISTIKA")
print("=" * 70)

numeric_cols = ["Sales", "Customers", "CompetitionDistance"]

print("\nOsnovna deskriptivna statistika (numericke varijable):")
print(df[numeric_cols].describe())

print("\nMjere oblika distribucije (skewness, kurtosis):")
for col in numeric_cols:
    sk = stats.skew(df[col].dropna())
    ku = stats.kurtosis(df[col].dropna())
    print(f"  {col}: skewness={sk:.3f}, kurtosis={ku:.3f}")

# Histogrami 
fig, axes = plt.subplots(1, len(numeric_cols), figsize=(6.5 * len(numeric_cols), 4.5))
fig.suptitle("Distribucija numerickih varijabli", fontsize=14, fontweight="bold")
for ax, col in zip(axes, numeric_cols):
    ax.hist(df[col], bins=50, color="#4C72B0", edgecolor="black", alpha=0.75)
    ax.axvline(df[col].mean(), color="red", linestyle="--",
               label=f"Mean: {df[col].mean():.1f}")
    ax.axvline(df[col].median(), color="green", linestyle="--",
               label=f"Median: {df[col].median():.1f}")
    ax.set_title(f"Distribucija - {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frekvencija")
    ax.legend()
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, "02_distribucija_numerickih.png"), dpi=120)
plt.close()

# Boxplotovi 
fig, axes = plt.subplots(1, len(numeric_cols), figsize=(4.5 * len(numeric_cols), 5))
fig.suptitle("Box plotovi - analiza outliera", fontsize=14, fontweight="bold")
for ax, col in zip(axes, numeric_cols):
    ax.boxplot(df[col].dropna())
    ax.set_title(col)
    ax.set_ylabel("Vrijednost")
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, "03_boxplot_outlieri.png"), dpi=120)
plt.close()

print("\nGrafici sacuvani: outputs/02_distribucija_numerickih.png, "
      "outputs/03_boxplot_outlieri.png")


# =================================================================
# 6. ANALIZA KATEGORICKIH VARIJABLI
# =================================================================
print("\n" + "=" * 70)
print("6. ANALIZA KATEGORICKIH VARIJABLI")
print("=" * 70)

cat_cols = ["StoreType", "Assortment", "StateHoliday", "Promo", "DayOfWeek"]
print("Distribucija kategorickih varijabli:")
for col in cat_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Distribucija kategorickih varijabli", fontsize=14, fontweight="bold")
axes = axes.flatten()
palette = sns.color_palette("tab10")
for i, col in enumerate(cat_cols):
    counts = df[col].astype(str).value_counts()
    axes[i].bar(counts.index, counts.values, color=palette[:len(counts)])
    for j, v in enumerate(counts.values):
        axes[i].text(j, v, f"{v}", ha="center", va="bottom", fontsize=9)
    axes[i].set_title(f"Distribucija - {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Broj redova")
for j in range(len(cat_cols), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, "04_kategoricke_distribucija.png"), dpi=120)
plt.close()
print("\nGraf sacuvan: outputs/04_kategoricke_distribucija.png")


# =================================================================
# 7. DISTRIBUCIJA CILJNE VARIJABLE (Sales)
# =================================================================
print("\n" + "=" * 70)
print("7. DISTRIBUCIJA CILJNE VARIJABLE (Sales)")
print("=" * 70)
# Sales je kontinuirana varijabla, pa se njena distribucija
# prikazuje histogramom i boxplotom.

print(df["Sales"].describe())
sk = stats.skew(df["Sales"])
print(f"Skewness Sales: {sk:.3f} (blago pozitivno asimetricna - ima dana "
      f"s izuzetno visokom prodajom)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Distribucija ciljne varijable - Sales", fontsize=14, fontweight="bold")
axes[0].hist(df["Sales"], bins=60, color="#C44E52", edgecolor="black", alpha=0.8)
axes[0].axvline(df["Sales"].mean(), color="black", linestyle="--",
                 label=f"Mean: {df['Sales'].mean():.0f}")
axes[0].axvline(df["Sales"].median(), color="blue", linestyle="--",
                 label=f"Median: {df['Sales'].median():.0f}")
axes[0].set_title("Histogram")
axes[0].set_xlabel("Sales")
axes[0].set_ylabel("Frekvencija")
axes[0].legend()

axes[1].boxplot(df["Sales"], vert=True)
axes[1].set_title("Boxplot")
axes[1].set_ylabel("Sales")

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, "05_target_distribucija.png"), dpi=120)
plt.close()
print("\nGraf sacuvan: outputs/05_target_distribucija.png")


# =================================================================
# 8. ANALIZA ODNOSA IZMEDJU VARIJABLI I CILJNE VARIJABLE
# =================================================================
print("\n" + "=" * 70)
print("8. ANALIZA ODNOSA IZMEDJU VARIJABLI I CILJNE VARIJABLE")
print("=" * 70)

# Numericke varijable vs Sales (hexbin gustine - preglednije od
# scatter plota za 800k+ tacaka)
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
fig.suptitle("Odnos numerickih varijabli i ciljne varijable (Sales)",
              fontsize=14, fontweight="bold")

hb1 = axes[0].hexbin(df["Customers"], df["Sales"], gridsize=50, cmap="Blues",
                      mincnt=1)
axes[0].set_title("Customers vs. Sales")
axes[0].set_xlabel("Customers")
axes[0].set_ylabel("Sales")
plt.colorbar(hb1, ax=axes[0], label="broj zapisa")

hb2 = axes[1].hexbin(df["CompetitionDistance"], df["Sales"], gridsize=50,
                      cmap="Greens", mincnt=1)
axes[1].set_title("CompetitionDistance vs. Sales")
axes[1].set_xlabel("CompetitionDistance")
axes[1].set_ylabel("Sales")
plt.colorbar(hb2, ax=axes[1], label="broj zapisa")

df.boxplot(column="Sales", by="DayOfWeek", ax=axes[2], grid=False)
axes[2].set_title("Sales po danu u sedmici")
axes[2].set_xlabel("DayOfWeek (1=Pon ... 7=Ned)")
axes[2].set_ylabel("Sales")
plt.suptitle("Odnos numerickih varijabli i ciljne varijable (Sales)",
              fontsize=14, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUT_DIR, "06_odnos_numericke_sales.png"), dpi=120)
plt.close()

print("Customers ima vidljivo najjaci (pozitivan) odnos sa Sales - vise "
      "kupaca gotovo linearno prati veca prodaja.")
print("Graf sacuvan: outputs/06_odnos_numericke_sales.png")

# Kategoricke varijable vs Sales (prosjecan Sales po kategoriji)
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Prosjecan Sales po kategorickim varijablama",
              fontsize=14, fontweight="bold")
axes = axes.flatten()
cat_vs_target = ["StoreType", "Assortment", "Promo", "StateHoliday"]
for i, col in enumerate(cat_vs_target):
    means = df.groupby(col)["Sales"].mean().sort_values(ascending=False)
    axes[i].bar(means.index.astype(str), means.values, color=palette[:len(means)])
    for j, v in enumerate(means.values):
        axes[i].text(j, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    axes[i].set_title(f"Prosjecan Sales po {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Prosjecan Sales")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, "07_odnos_kategoricke_sales.png"), dpi=120)
plt.close()
print("Graf sacuvan: outputs/07_odnos_kategoricke_sales.png")


# =================================================================
# 9. ENKODIRANJE KATEGORICKIH VARIJABLI
# =================================================================
print("\n" + "=" * 70)
print("9. ENKODIRANJE KATEGORICKIH VARIJABLI")
print("=" * 70)

encode_cols = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]
encoders = {}
for col in encode_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f"{col} -> {col}_enc  (klase: {list(le.classes_)})")


# =================================================================
# 10. KORELACIJSKA ANALIZA
# =================================================================
print("\n" + "=" * 70)
print("10. KORELACIJSKA ANALIZA")
print("=" * 70)

corr_features = [
    "Sales", "Customers", "DayOfWeek", "Promo", "SchoolHoliday",
    "CompetitionDistance", "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek",
    "Promo2SinceYear", "Year", "Month", "Day", "WeekOfYear",
    "StoreType_enc", "Assortment_enc", "StateHoliday_enc",
    "PromoInterval_enc",
]

corr_matrix = df[corr_features].corr()

# Triangularna maska
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, annot_kws={"size": 7}, square=True,
            cbar_kws={"shrink": 0.8})
plt.title("Korelacijska matrica - Rossmann Store Sales", fontsize=14,
          fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "08_korelaciona_matrica.png"), dpi=120)
plt.close()

print("Korelacije sa ciljnom varijablom Sales (sortirano):")
sales_corr = corr_matrix["Sales"].drop("Sales").sort_values(key=abs, ascending=False)
print(sales_corr)

THRESHOLD = 0.75
print(f"\nParovi featura (bez Sales) s |korelacijom| > {THRESHOLD}:")
feat_corr = corr_matrix.drop(index="Sales", columns="Sales")
high_corr_pairs = []
for i in range(len(feat_corr.columns)):
    for j in range(i + 1, len(feat_corr.columns)):
        val = feat_corr.iloc[i, j]
        if abs(val) > THRESHOLD:
            c1, c2 = feat_corr.columns[i], feat_corr.columns[j]
            print(f"  {c1} <-> {c2}: {val:.3f}")
            high_corr_pairs.append((c1, c2, val))
if not high_corr_pairs:
    print("  (nema parova iznad praga)")

print("\nGraf sacuvan: outputs/08_korelaciona_matrica.png")


# =================================================================
# 11. PREPROCESSING - DEFINISANJE X, y I REDUCIRANOG SKUPA
# =================================================================
print("\n" + "=" * 70)
print("11. PREPROCESSING PODATAKA")
print("=" * 70)

# NAPOMENA: Customers ima ocekivano vrlo jaku korelaciju sa Sales, ali
# u realnoj primjeni (predikcija BUDUCE prodaje) broj kupaca tog dana
# jos nije poznat unaprijed - to je "curenje informacije" (data leakage).
# Zadrzana je u modelu, ali je to navedeno kao ogranicenje rada.

all_features = [
    "DayOfWeek", "Customers", "Promo", "SchoolHoliday",
    "CompetitionDistance", "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek",
    "Promo2SinceYear", "Year", "Month", "Day", "WeekOfYear",
    "StoreType_enc", "Assortment_enc", "StateHoliday_enc",
    "PromoInterval_enc",
]

to_drop = set()
for c1, c2, val in high_corr_pairs:
    corr1 = abs(sales_corr.get(c1, 0))
    corr2 = abs(sales_corr.get(c2, 0))
    weaker = c2 if corr1 >= corr2 else c1
    to_drop.add(weaker)

reduced_features = [f for f in all_features if f not in to_drop]
print(f"Varijable izbacene iz reduciranog skupa (multikolinearnost): "
      f"{sorted(to_drop) if to_drop else 'nijedna'}")
print(f"Sve varijable ({len(all_features)}): {all_features}")
print(f"Reducirane varijable ({len(reduced_features)}): {reduced_features}")

X_all = df[all_features].copy()
X_reduced = df[reduced_features].copy()
y = df["Sales"].copy()

if SAMPLE_SIZE is not None and len(df) > SAMPLE_SIZE:
    idx = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).index
    X_all = X_all.loc[idx]
    X_reduced = X_reduced.loc[idx]
    y = y.loc[idx]
    print(f"\nKoristi se uzorak od {SAMPLE_SIZE} redova (od ukupno {len(df)})")


# =================================================================
# 12. MODELIRANJE - ITERACIJE
# =================================================================
print("\n" + "=" * 70)
print("12. MODELIRANJE - ITERACIJE")
print("=" * 70)


def train_and_evaluate(X, y, test_size, model, scale=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "model": model,
            "y_test": y_test, "y_pred": y_pred}


iterations = [
    ("Iter 1: Linearna regresija, sve varijable, 80:20",
     X_all, 0.20, LinearRegression(), "Linearna regresija", True),
    ("Iter 2: Linearna regresija, reducirane varijable, 80:20",
     X_reduced, 0.20, LinearRegression(), "Linearna regresija", True),
    ("Iter 3: Linearna regresija, sve varijable, 70:30",
     X_all, 0.30, LinearRegression(), "Linearna regresija", True),
    ("Iter 4: Random Forest, sve varijable, 80:20",
     X_all, 0.20, RandomForestRegressor(
         n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1),
     "Random Forest", False),
    ("Iter 5: Random Forest, reducirane varijable, 80:20",
     X_reduced, 0.20, RandomForestRegressor(
         n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1),
     "Random Forest", False),
    ("Iter 6: Random Forest, sve varijable, 70:30",
     X_all, 0.30, RandomForestRegressor(
         n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1),
     "Random Forest", False),
]

results = []
fitted = {}
for name, X, test_size, model, model_name, scale in iterations:
    print(f"\n--- {name} ---")
    res = train_and_evaluate(X, y, test_size, model, scale)
    print(f"RMSE={res['RMSE']:.2f}  MAE={res['MAE']:.2f}  R2={res['R2']:.4f}")
    results.append({
        "Iteracija": name,
        "Algoritam": model_name,
        "Broj varijabli": X.shape[1],
        "Omjer": f"{int((1-test_size)*100)}:{int(test_size*100)}",
        "RMSE": round(res["RMSE"], 2),
        "MAE": round(res["MAE"], 2),
        "R2": round(res["R2"], 4),
    })
    fitted[name] = res


# =================================================================
# 13. ANALIZA REZULTATA
# =================================================================
print("\n" + "=" * 70)
print("13. ANALIZA REZULTATA")
print("=" * 70)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(OUT_DIR, "rezultati_iteracija.csv"), index=False)

best_row = results_df.loc[results_df["R2"].idxmax()]
print(f"\nNajbolji rezultat po R2: {best_row['Iteracija']} "
      f"(R2={best_row['R2']}, RMSE={best_row['RMSE']})")

# --- Actual vs Predicted (regresijski ekvivalent matrice konfuzije/ROC) ---
lr_res = fitted["Iter 1: Linearna regresija, sve varijable, 80:20"]
rf_res = fitted["Iter 4: Random Forest, sve varijable, 80:20"]

# Uzorak od 3000 tacaka za citljiv scatter (na punom setu bi bilo prezguzano)
rng = np.random.RandomState(RANDOM_STATE)


def sample_for_plot(y_test, y_pred, n=3000):
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    if len(y_test) > n:
        idx = rng.choice(len(y_test), n, replace=False)
        return y_test[idx], y_pred[idx]
    return y_test, y_pred


fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
fig.suptitle("Actual vs. Predicted - Linearna regresija vs. Random Forest "
             "(sve varijable, 80:20)", fontsize=13, fontweight="bold")

for ax, res, name, color in [
    (axes[0], lr_res, "Linearna regresija", COLOR_LR),
    (axes[1], rf_res, "Random Forest", COLOR_RF),
]:
    yt, yp = sample_for_plot(res["y_test"], res["y_pred"])
    ax.scatter(yt, yp, alpha=0.3, s=12, color=color)
    lims = [0, max(yt.max(), yp.max())]
    ax.plot(lims, lims, "k--", label="Savrsena predikcija")
    ax.set_xlabel("Stvarni Sales")
    ax.set_ylabel("Predvidjeni Sales")
    ax.set_title(f"{name}  (R2={r2_score(res['y_test'], res['y_pred']):.3f})")
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUT_DIR, "09_actual_vs_predicted.png"), dpi=120)
plt.close()
print("\nGraf sacuvan: outputs/09_actual_vs_predicted.png")

# Usporedba RMSE/R2 po iteracijama (boja po algoritmu)
bar_colors = [COLOR_LR if a == "Linearna regresija" else COLOR_RF
              for a in results_df["Algoritam"]]
labels = [f"Iter {i+1}" for i in range(len(results_df))]

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
fig.suptitle("Usporedba modela - sve iteracije", fontsize=14, fontweight="bold")

axes[0].bar(labels, results_df["RMSE"], color=bar_colors)
for i, v in enumerate(results_df["RMSE"]):
    axes[0].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
axes[0].set_title("RMSE po iteracijama (nize = bolje)")
axes[0].set_ylabel("RMSE")
axes[0].tick_params(axis="x", rotation=30)

axes[1].bar(labels, results_df["R2"], color=bar_colors)
for i, v in enumerate(results_df["R2"]):
    axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
axes[1].set_title("R2 po iteracijama (vise = bolje)")
axes[1].set_ylabel("R2")
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis="x", rotation=30)

lr_patch = mpatches.Patch(color=COLOR_LR, label="Linearna regresija")
rf_patch = mpatches.Patch(color=COLOR_RF, label="Random Forest")
fig.legend(handles=[lr_patch, rf_patch], loc="upper right",
           bbox_to_anchor=(0.99, 0.97))

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUT_DIR, "10_usporedba_iteracija.png"), dpi=120)
plt.close()
print("Graf sacuvan: outputs/10_usporedba_iteracija.png")

# Vaznost varijabli (najvaznija istaknuta crvenom bojom)
best_rf = fitted["Iter 6: Random Forest, sve varijable, 70:30"]["model"]
importances = pd.Series(best_rf.feature_importances_, index=X_all.columns)
importances = importances.sort_values(ascending=True)  # ascending za barh

colors_imp = ["#4C72B0"] * len(importances)
colors_imp[-1] = "crimson"  # najvaznija varijabla (posljednja nakon sortiranja asc)

plt.figure(figsize=(10, 7))
plt.barh(importances.index, importances.values, color=colors_imp)
plt.title("Vaznost varijabli - Random Forest\n(crveno = najvaznija varijabla)",
          fontsize=13)
plt.xlabel("Vaznost varijable")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "11_vaznost_varijabli.png"), dpi=120)
plt.close()

print("\nVaznost varijabli (Random Forest, opadajuce):")
print(importances.sort_values(ascending=False))
print("\nGraf sacuvan: outputs/11_vaznost_varijabli.png")

print("\n" + "=" * 70)
print(f"Svi grafici i rezultati su u folderu: {OUT_DIR}/")
print("Gotovo.")
print("=" * 70)
