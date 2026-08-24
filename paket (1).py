#untuk import library
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    validation_curve,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay
)

#untuk setting visualisasi
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (9, 5)
pd.set_option("display.max_columns", 50)

RANDOM_STATE = 42

# ======================================================================
# EDA
# ======================================================================

df = pd.read_csv(
    "DataCoSupplyChainDataset.csv",
    sep=";",
    decimal=",",
    encoding="latin1"
)

df.rename(columns=lambda x: x.replace("ï»¿", ""), inplace=True)
df.shape

df.shape
print("Jumlah data: ", df.shape)

df.info()

df.describe()

df.isnull().sum()

missing = pd.DataFrame({
    'Jumlah Missing': df.isnull().sum(),
    'Persentase (%)': round(df.isnull().mean() * 100, 2)
})

missing[missing['Jumlah Missing'] > 0]

# Menghapus kolom dengan missing value tinggi
df.drop(columns=['Product Description', 'Order Zipcode'], inplace=True)

# Mengisi missing value
df['Customer Lname'].fillna(df['Customer Lname'].mode()[0], inplace=True)
df['Customer Zipcode'].fillna(df['Customer Zipcode'].mode()[0], inplace=True)

# Mengecek missing value setelah preprocessing
df.isnull().sum()

print("Modus Customer Zip code:")
print(df["Customer Zipcode"].mode())

print("\nModus Customer Lname:")
print(df["Customer Lname"].mode())

# Mengecek & menghapus data duplikat
duplicates = df.duplicated().sum()
print(f"Jumlah data duplikat sebelum dihapus: {duplicates}")

df = df.drop_duplicates().reset_index(drop=True)

print(f"Jumlah data duplikat setelah dihapus: {df.duplicated().sum()}")
print("Shape df setelah hapus duplikat:", df.shape)

#ini data yg udh dibersihin (kolomnya sisa 51)
df.shape

# Distribusi target
print(df['Late_delivery_risk'].value_counts())

print((df['Late_delivery_risk'].value_counts(normalize=True) * 100).round(2))

plt.figure(figsize=(5,4))
sns.countplot(data=df, x='Late_delivery_risk', palette='Set2')
plt.title('Distribusi Target: Late_delivery_risk')
plt.xlabel('0 = Tidak Berisiko Terlambat, 1 = Terlambat')
plt.ylabel('Jumlah Data')
plt.show()

kolom_numerik = [
    'Sales',
    'Benefit per order',
    'Order Item Product Price',
    'Product Price',
    'Order Item Quantity',
    'Days for shipping (real)'
]

plt.figure(figsize=(15,8))

for i, col in enumerate(kolom_numerik, 1):
    plt.subplot(2,3,i)
    plt.hist(df[col], bins=30)
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('Frekuensi')

plt.tight_layout()
plt.show()

kolom = [
    'Benefit per order',
    'Sales per customer',
    'Order Item Discount',
    'Order Item Product Price',
    'Product Price',
    'Sales',
    'Order Item Quantity',
    'Days for shipping (real)',
    'Days for shipment (scheduled)'
]

plt.figure(figsize=(15, 10))

for i, col in enumerate(kolom, 1):
    plt.subplot(3, 3, i)
    plt.boxplot(df[col].dropna(), vert=False)
    plt.title(col, fontsize=10)

plt.tight_layout()
plt.show()

kolom_kategori = [
    'Shipping Mode',
    'Market',
    'Order Region',
    'Category Name'
]

plt.figure(figsize=(16,12))

for i, col in enumerate(kolom_kategori, 1):
    plt.subplot(2,2,i)

    urutan = df[col].value_counts().index

    sns.countplot(
        data=df,
        y=col,
        order=urutan,
        palette='Set2'
    )

    plt.title(f'Distribusi {col}')
    plt.xlabel('Jumlah')
    plt.ylabel('')

plt.tight_layout()
plt.show()

kolom = [
    'Days for shipping (real)',
    'Days for shipment (scheduled)',
    'Sales',
    'Sales per customer',
    'Benefit per order',
    'Order Item Product Price',
    'Product Price',
    'Order Item Discount',
    'Order Item Quantity',
    'Late_delivery_risk'   # target
]

plt.figure(figsize=(10,8))

sns.heatmap(
    df[kolom].corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)

plt.title('Heatmap Korelasi Fitur dengan Target')
plt.show()

mode_col = df['Shipping Mode']

mode_risk = df.groupby('Shipping Mode')['Late_delivery_risk'].mean().sort_values(ascending=False) * 100

# Otomatis pakai nama asli kalau kolom masih teks, atau kode angka kalau udah di-encode
if mode_col.dtype == 'object':
    labels_mode = mode_risk.index.astype(str)
else:
    labels_mode = mode_risk.index.astype(str)  # tampil sebagai kode angka saja

fig, ax = plt.subplots(figsize=(9,5))
colors = ['#FF5C5C' if v > mode_risk.mean() else '#00C2A8' for v in mode_risk]
bars = ax.bar(labels_mode, mode_risk, color=colors)
ax.bar_label(bars, fmt='%.1f%%', padding=3)
ax.axhline(mode_risk.mean(), ls='--', color='gray', label=f'Rata-rata: {mode_risk.mean():.1f}%')
ax.set_title('Late Delivery Rate (%) by Shipping Mode')
ax.set_ylabel('Late Delivery Rate (%)')
plt.xticks(rotation=20, ha='right')
ax.legend()
plt.tight_layout()
plt.show()

market_risk = df.groupby('Market')['Late_delivery_risk'].mean().sort_values(ascending=False) * 100
labels_market = market_risk.index.astype(str)

fig, ax = plt.subplots(figsize=(9,5))
colors = ['#FF5C5C' if v > market_risk.mean() else '#00C2A8' for v in market_risk]
bars = ax.bar(labels_market, market_risk, color=colors)
ax.bar_label(bars, fmt='%.1f%%', padding=3)
ax.axhline(market_risk.mean(), ls='--', color='gray', label=f'Rata-rata: {market_risk.mean():.1f}%')
ax.set_title('Late Delivery Rate (%) by Market')
ax.set_ylabel('Late Delivery Rate (%)')
plt.xticks(rotation=20, ha='right')
ax.legend()
plt.tight_layout()
plt.show()

cat_counts = df['Category Name'].value_counts()
valid_cats = cat_counts[cat_counts >= 30].index

cat_risk = (df[df['Category Name'].isin(valid_cats)]
            .groupby('Category Name')['Late_delivery_risk']
            .mean().sort_values(ascending=False).head(10) * 100)

fig, ax = plt.subplots(figsize=(9,6))
bars = ax.barh(cat_risk.index.astype(str), cat_risk, color='#FF5C5C')
ax.bar_label(bars, fmt='%.1f%%', padding=3)
ax.invert_yaxis()
ax.set_title('Top 10 Kategori Produk dengan Late Delivery Rate Tertinggi')
ax.set_xlabel('Late Delivery Rate (%)')
plt.tight_layout()
plt.show()

target_corr = (df.select_dtypes(include=np.number)
               .corr()['Late_delivery_risk']
               .drop('Late_delivery_risk'))
target_corr = target_corr.reindex(target_corr.abs().sort_values(ascending=False).index).head(10)

fig, ax = plt.subplots(figsize=(8,6))
colors = ['#FF5C5C' if v > 0 else '#00C2A8' for v in target_corr]
bars = ax.barh(target_corr.index, target_corr, color=colors)
ax.bar_label(bars, fmt='%.2f', padding=3)
ax.invert_yaxis()
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Top 10 Fitur dengan Korelasi Tertinggi ke Late_delivery_risk')
ax.set_xlabel('Korelasi (Pearson)')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7,5))
sns.boxplot(data=df, x='Late_delivery_risk', y='Days for shipment (scheduled)',
            palette=['#00C2A8', '#FF5C5C'], ax=ax)
ax.set_xticklabels(['Tidak Terlambat', 'Terlambat'])
ax.set_title('Days for shipment (scheduled) vs Late Delivery Risk')
plt.tight_layout()
plt.show()

df_time = df.copy()
df_time['order_month'] = pd.to_datetime(df_time['order date (DateOrders)']).dt.to_period('M')

monthly_risk = df_time.groupby('order_month')['Late_delivery_risk'].mean() * 100

fig, ax = plt.subplots(figsize=(12,5))
monthly_risk.plot(ax=ax, marker='o', color='#FF5C5C')
ax.axhline(monthly_risk.mean(), ls='--', color='gray', label=f'Rata-rata: {monthly_risk.mean():.1f}%')
ax.set_title('Tren Late Delivery Rate per Bulan')
ax.set_ylabel('Late Delivery Rate (%)')
ax.set_xlabel('Bulan')
ax.legend()
plt.tight_layout()
plt.show()

# ======================================================================
# Data Preprocessing
# ======================================================================

# Menghapus fitur identitas, ID yang tidak diperlukan, dan fitur yang redundant
df.drop(columns=[
    'Customer Email',
    'Customer Password',
    'Customer Fname',
    'Customer Lname',
    'Customer Street',
    'Customer Id',
    'Order Customer Id',
    'Order Id',
    'Order Item Id',
    'Order Item Cardprod Id',
    'Product Card Id',
    'Product Image',
    'Product Status',
    'Category Id',
    'Department Id',
    'Product Price',
    'Order Profit Per Order',
    'order date (DateOrders)',
    'shipping date (DateOrders)',
    'Days for shipping (real)',
    'Delivery Status',
], inplace=True)

print(f"Jumlah kolom setelah preprocessing: {df.shape[1]}")
df.info()

#encoding ubah kategorikal/object jd numerik
label_encoders = {}

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

target = "Late_delivery_risk"

cols = [c for c in df.columns if c != target] + [target]
df = df[cols]
plt.figure(figsize=(20,16))

sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0
)

plt.title("Heatmap Korelasi Antar Fitur")
plt.show()

mapping = pd.DataFrame({
    "Kategori Asli": label_encoders["Type"].classes_,
    "Hasil Encoding": range(len(label_encoders["Type"].classes_))
})

print(mapping)

# ======================================================================
# # Feature Engineering
# ======================================================================

# fitur baru: Total_Item_Value (harga x qty) & High_Discount_Flag (diskon di atas rata2 apa enggak)
#'Days for shipping (real)' gapake krn udh didrop dari awal - itu langsung nentuin telat/egknya (leakage kalo dipake)

df['Total_Item_Value'] = df['Order Item Product Price'] * df['Order Item Quantity']
df['High_Discount_Flag'] = (df['Order Item Discount Rate'] > df['Order Item Discount Rate'].median()).astype(int)

print("Fitur baru berhasil dibuat: 'Total_Item_Value' dan 'High_Discount_Flag'")
df[['Order Item Product Price', 'Order Item Quantity', 'Total_Item_Value',
    'Order Item Discount Rate', 'High_Discount_Flag']].head(10)

# Split data

SAMPLE_SIZE = 40000

# 1. Sampling 40rb data
if len(df) > SAMPLE_SIZE:
    df, _ = train_test_split(
        df,
        train_size=SAMPLE_SIZE,
        stratify=df["Late_delivery_risk"],
        random_state=42
    )

print(df.shape)

print(df["Late_delivery_risk"].value_counts())

print((df["Late_delivery_risk"].value_counts(normalize=True) * 100).round(2))

# Grafik distribusi target
plt.figure(figsize=(5,4))
sns.countplot(data=df, x='Late_delivery_risk', palette='Set2')
plt.title('Distribusi Target Setelah Sampling')
plt.xlabel('0 = Tidak Berisiko Terlambat, 1 = Terlambat')
plt.ylabel('Jumlah Data')
plt.show()

# 2. Untuk membuat fitur dan target
X = df.drop("Late_delivery_risk", axis=1)
y = df["Late_delivery_risk"]

print(X.shape)
print(len(X.columns))

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Jumlah data training:", X_train.shape)
print("Jumlah data testing :", X_test.shape)

# ======================================================================
# # Feature Selection
# ======================================================================

# Feature Selection: fit HANYA pada data training untuk menghindari data leakage,
# baru diterapkan (transform) ke data training & testing dengan fitur yang sama

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

results = []

for k in [10, 15, 17, 20, 30, 31]:

    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_fs = selector.fit_transform(X_train, y_train)
    X_test_fs = selector.transform(X_test)

    model = XGBClassifier(
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train_fs, y_train)

    y_prob = model.predict_proba(X_test_fs)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    results.append((k, auc))

for k, auc in results:
    print(f"k = {k:2d} | ROC-AUC = {auc:.4f}")

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=15)
selector.fit(X_train, y_train)

selected_features = X_train.columns[selector.get_support()]
feature_table = pd.DataFrame({
    "No": range(1, len(selected_features) + 1),
    "Selected Feature": selected_features
})
print(feature_table)

X_train = X_train[selected_features]
X_test = X_test[selected_features]

print(f"\nShape X_train : {X_train.shape}")
print(f"Shape X_test  : {X_test.shape}")

# ======================================================================
# ALGORITMA LOGISTIC REGRESSION
# ======================================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# BASELINE LOGISTIC REGRESSION
# ==========================================

lr_base = LogisticRegression(
    random_state=42,
    max_iter=1000
)

lr_base.fit(X_train_scaled, y_train)

y_pred_lr_base = lr_base.predict(X_test_scaled)
y_prob_lr_base = lr_base.predict_proba(X_test_scaled)[:, 1]

print("=== Baseline Logistic Regression ===")
print("Accuracy :", accuracy_score(y_test, y_pred_lr_base))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob_lr_base))
print(classification_report(y_test, y_pred_lr_base))

cm = confusion_matrix(y_test, y_pred_lr_base)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Tidak Terlambat', 'Terlambat'],
    yticklabels=['Tidak Terlambat', 'Terlambat']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

train_acc = accuracy_score(y_train, lr_base.predict(X_train_scaled)) * 100
test_acc = accuracy_score(y_test, lr_base.predict(X_test_scaled)) * 100

print(f"Training Accuracy : {train_acc:.2f}%")
print(f"Testing Accuracy  : {test_acc:.2f}%")
print(f"Gap               : {train_acc - test_acc:.2f}%")

# Probabilitas kelas 1 (Late Delivery)
y_prob = lr_base.predict_proba(X_test_scaled)[:, 1]

plt.figure(figsize=(8,5))
plt.hist(y_prob, bins=30, edgecolor='black')
plt.axvline(0.5, color='red', linestyle='--', label='Threshold = 0.5')

plt.title('Distribusi Probabilitas Prediksi Logistic Regression')
plt.xlabel('Probabilitas Late Delivery')
plt.ylabel('Jumlah Data')
plt.legend()
plt.show()

# Nilai linear (z) dari model
z = lr_base.decision_function(X_test_scaled)

# Probabilitas hasil model
prob = lr_base.predict_proba(X_test_scaled)[:, 1]

# Urutkan agar membentuk kurva
idx = np.argsort(z)
z_sorted = z[idx]
prob_sorted = prob[idx]

plt.figure(figsize=(8,5))
plt.plot(z_sorted, prob_sorted, color='blue', linewidth=2)

plt.axhline(0.5, color='red', linestyle='--', label='Threshold = 0.5')
plt.axvline(0, color='gray', linestyle='--')

plt.xlabel('Nilai Linear (z)')
plt.ylabel('Probabilitas')
plt.title('Kurva Sigmoid Logistic Regression')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("\n" + "="*60)
print("==== HYPERPARAMETER TUNING LOGISTIC REGRESSION ====")
print("="*60 + "\n")

param_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs'],
    'penalty': ['l2'],
    'class_weight': [None, 'balanced']
}

grid_lr = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid=param_lr,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_lr.fit(X_train_scaled, y_train)

print("LR - Best Parameter:", grid_lr.best_params_)
print("LR - Best ROC-AUC  :", grid_lr.best_score_)

lr = grid_lr.best_estimator_

y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

print("Accuracy :", accuracy_score(y_test, y_pred_lr))
print("\nClassification Report")
print(classification_report(y_test, y_pred_lr))

train_acc = accuracy_score(y_train, lr.predict(X_train_scaled)) * 100
test_acc = accuracy_score(y_test, lr.predict(X_test_scaled)) * 100

print(f"Training Accuracy : {train_acc:.2f}%")
print(f"Testing Accuracy  : {test_acc:.2f}%")
print(f"Gap               : {train_acc - test_acc:.2f}%")

cm = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Tidak Terlambat', 'Terlambat'],
    yticklabels=['Tidak Terlambat', 'Terlambat']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# Probabilitas kelas 1 (Late Delivery)
y_prob = lr.predict_proba(X_test_scaled)[:, 1]

plt.figure(figsize=(8,5))
plt.hist(y_prob, bins=30, edgecolor='black')
plt.axvline(0.5, color='red', linestyle='--', label='Threshold = 0.5')

plt.title('Distribusi Probabilitas Prediksi Logistic Regression')
plt.xlabel('Probabilitas Late Delivery')
plt.ylabel('Jumlah Data')
plt.legend()
plt.show()

# Nilai linear (z) dari model
z = lr.decision_function(X_test_scaled)

# Probabilitas hasil model
prob = lr.predict_proba(X_test_scaled)[:, 1]

# Urutkan agar membentuk kurva
idx = np.argsort(z)
z_sorted = z[idx]
prob_sorted = prob[idx]

plt.figure(figsize=(8,5))
plt.plot(z_sorted, prob_sorted, color='blue', linewidth=2)

plt.axhline(0.5, color='red', linestyle='--', label='Threshold = 0.5')
plt.axvline(0, color='gray', linestyle='--')

plt.xlabel('Nilai Linear (z)')
plt.ylabel('Probabilitas')
plt.title('Kurva Sigmoid Logistic Regression')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, y_prob)

# ======================================================================
# ALGORITMA RANDOM FOREST
# ======================================================================

# ==========================
# BASELINE RANDOM FOREST
# ==========================

rf_base = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

rf_base.fit(X_train, y_train)

y_pred_rf_base = rf_base.predict(X_test)
y_prob_rf_base = rf_base.predict_proba(X_test)[:, 1]

print("=== Baseline Random Forest ===")
print("Accuracy :", accuracy_score(y_test, y_pred_rf_base))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob_rf_base))
print(classification_report(y_test, y_pred_rf_base))

cm = confusion_matrix(y_test, y_pred_rf_base)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=['Tidak Terlambat', 'Terlambat'],
    yticklabels=['Tidak Terlambat', 'Terlambat']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

train_pred = rf_base.predict(X_train)

train_acc = accuracy_score(y_train, train_pred) * 100
test_acc = accuracy_score(y_test, y_pred_rf_base) * 100

print(f"Training Accuracy : {train_acc:.2f}%")
print(f"Testing Accuracy  : {test_acc:.2f}%")
print(f"Gap               : {train_acc - test_acc:.2f}%")

#visualisasi rf
from sklearn.tree import plot_tree
plt.figure(figsize=(22, 10))

plot_tree(
    rf_base.estimators_[0],      
    feature_names=X.columns,
    class_names=["Tidak Terlambat", "Terlambat"],
    filled=True,
    rounded=True,
    max_depth=4,            
    fontsize=8
)

plt.title("Visualisasi Salah Satu Pohon pada Random Forest")
plt.show()

print("\n" + "="*60)
print("==== HYPERPARAMETER TUNING RANDOM FOREST ====")
print("="*60 + "\n")

param_rf = {
    'n_estimators': [100, 200],
    'max_depth': [8, 12, 16],
    'min_samples_split': [20, 40, 60],
    'min_samples_leaf': [10, 20, 30]
}

random_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_rf,
    n_iter=15, cv=5, scoring='roc_auc', n_jobs=-1, random_state=42
)
random_rf.fit(X_train, y_train)

print("RF - Best Parameter:", random_rf.best_params_)
print("RF - Best ROC-AUC  :", random_rf.best_score_)

rf = random_rf.best_estimator_   

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=['Tidak Terlambat', 'Terlambat'],
    yticklabels=['Tidak Terlambat', 'Terlambat']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

train_pred = rf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred) * 100
test_acc = accuracy_score(y_test, y_pred_rf) * 100

print(f"Training Accuracy : {train_acc:.2f}%")
print(f"Testing Accuracy  : {test_acc:.2f}%")
print(f"Gap               : {train_acc - test_acc:.2f}%")

#visualisasi rf
from sklearn.tree import plot_tree
plt.figure(figsize=(22, 10))

plot_tree(
    rf.estimators_[0],      
    feature_names=X.columns,
    class_names=["Tidak Terlambat", "Terlambat"],
    filled=True,
    rounded=True,
    max_depth=4,            
    fontsize=8
)

plt.title("Visualisasi Salah Satu Pohon pada Random Forest")
plt.show()

# ======================================================================
# ALGORITMA XGBoost
# ======================================================================

# ==========================================
# BASELINE XGBOOST
# ==========================================

from xgboost import XGBClassifier

xgb_base = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)

xgb_base.fit(X_train, y_train)

y_pred_xgb_base = xgb_base.predict(X_test)
y_prob_xgb_base = xgb_base.predict_proba(X_test)[:, 1]

print("=== Baseline XGBoost ===")
print("Accuracy :", accuracy_score(y_test, y_pred_xgb_base))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob_xgb_base))
print(classification_report(y_test, y_pred_xgb_base))

cm = confusion_matrix(y_test, y_pred_xgb_base)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Purples',
    xticklabels=['Tidak Terlambat', 'Terlambat'],
    yticklabels=['Tidak Terlambat', 'Terlambat']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.show()

train_pred = xgb_base.predict(X_train)

train_acc = accuracy_score(y_train, train_pred) * 100
test_acc = accuracy_score(y_test, y_pred_xgb_base) * 100

print(f"Training Accuracy : {train_acc:.2f}%")
print(f"Testing Accuracy  : {test_acc:.2f}%")
print(f"Gap               : {train_acc - test_acc:.2f}%")

from xgboost import XGBClassifier

print("\n" + "="*60)
print("==== HYPERPARAMETER TUNING XGBOOST ====")
print("="*60 + "\n")

param_xgb = {
    'n_estimators': [200, 300, 500],
    'learning_rate': [0.03, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_child_weight': [5, 7, 10],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.3],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 3, 5]
}

random_xgb = RandomizedSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
    param_distributions=param_xgb,
    n_iter=50   , cv=5, scoring='roc_auc', n_jobs=-1, random_state=42
)
random_xgb.fit(X_train, y_train)

print("XGB - Best Parameter:", random_xgb.best_params_)
print("XGB - Best ROC-AUC  :", random_xgb.best_score_)

xgb = random_xgb.best_estimator_

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

print("Accuracy :", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

cm = confusion_matrix(y_test, y_pred_xgb)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Purples',
    xticklabels=['Tidak Terlambat', 'Terlambat'],
    yticklabels=['Tidak Terlambat', 'Terlambat']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.show()

train_pred = xgb.predict(X_train)

train_acc = accuracy_score(y_train, train_pred) * 100
test_acc = accuracy_score(y_test, y_pred_xgb) * 100

print(f"Training Accuracy : {train_acc:.2f}%")
print(f"Testing Accuracy  : {test_acc:.2f}%")
print(f"Gap               : {train_acc - test_acc:.2f}%")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

hasil = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],

    'Accuracy Baseline': [
        accuracy_score(y_test, y_pred_lr_base),
        accuracy_score(y_test, y_pred_rf_base),
        accuracy_score(y_test, y_pred_xgb_base)
    ],

    'Accuracy Tuning': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_xgb)
    ],

    'Precision Baseline': [
        precision_score(y_test, y_pred_lr_base),
        precision_score(y_test, y_pred_rf_base),
        precision_score(y_test, y_pred_xgb_base)
    ],

    'Precision Tuning': [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_xgb)
    ],

    'Recall Baseline': [
        recall_score(y_test, y_pred_lr_base),
        recall_score(y_test, y_pred_rf_base),
        recall_score(y_test, y_pred_xgb_base)
    ],

    'Recall Tuning': [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_xgb)
    ],

    'F1 Baseline': [
        f1_score(y_test, y_pred_lr_base),
        f1_score(y_test, y_pred_rf_base),
        f1_score(y_test, y_pred_xgb_base)
    ],

    'F1 Tuning': [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_xgb)
    ],

    'ROC-AUC Baseline': [
        roc_auc_score(y_test, y_prob_lr_base),
        roc_auc_score(y_test, y_prob_rf_base),
        roc_auc_score(y_test, y_prob_xgb_base)
    ],

    'ROC-AUC Tuning': [
        roc_auc_score(y_test, y_prob_lr),
        roc_auc_score(y_test, y_prob_rf),
        roc_auc_score(y_test, y_prob_xgb)
    ]
})

hasil = hasil.round(4)

print(hasil)

metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']

fig, axes = plt.subplots(2, 3, figsize=(15,8))
axes = axes.flatten()

for i, metric in enumerate(metrics):

    baseline = hasil[f'{metric} Baseline']
    tuning = hasil[f'{metric} Tuning']

    x = np.arange(len(hasil))
    width = 0.35

    axes[i].bar(x-width/2, baseline, width, label='Baseline')
    axes[i].bar(x+width/2, tuning, width, label='Tuning')

    axes[i].set_xticks(x)
    axes[i].set_xticklabels(hasil['Model'], rotation=20)
    axes[i].set_title(metric)
    axes[i].set_ylim(0,1)

    for j,v in enumerate(baseline):
        axes[i].text(j-width/2, v+0.01, f'{v:.3f}', ha='center', fontsize=8)

    for j,v in enumerate(tuning):
        axes[i].text(j+width/2, v+0.01, f'{v:.3f}', ha='center', fontsize=8)

axes[0].legend()
axes[-1].axis('off')

plt.tight_layout()
plt.show()

# ======================================================================
# PERBANDINGAN METRIK: BASELINE VS HYPERPARAMETER TUNING (5 METRIK)
# ======================================================================

algo_names = ['Logistic Regression', 'Random Forest', 'XGBoost']

# Prediksi & probabilitas baseline
preds_base  = [y_pred_lr_base, y_pred_rf_base, y_pred_xgb_base]
probs_base  = [y_prob_lr_base, y_prob_rf_base, y_prob_xgb_base]

# Prediksi & probabilitas hasil tuning
preds_tuned = [y_pred_lr, y_pred_rf, y_pred_xgb]
probs_tuned = [y_prob_lr, y_prob_rf, y_prob_xgb]

metrics = {
    'Accuracy':  lambda y_true, y_pred, y_prob: accuracy_score(y_true, y_pred),
    'Precision': lambda y_true, y_pred, y_prob: precision_score(y_true, y_pred),
    'Recall':    lambda y_true, y_pred, y_prob: recall_score(y_true, y_pred),
    'F1':        lambda y_true, y_pred, y_prob: f1_score(y_true, y_pred),
    'ROC-AUC':   lambda y_true, y_pred, y_prob: roc_auc_score(y_true, y_prob),
}

# Hitung tiap metrik untuk baseline & tuning, per algoritma
results = {}
for metric_name, func in metrics.items():
    base_vals  = [func(y_test, p, pr) for p, pr in zip(preds_base, probs_base)]
    tuned_vals = [func(y_test, p, pr) for p, pr in zip(preds_tuned, probs_tuned)]
    results[metric_name] = {'Baseline': base_vals, 'Tuning': tuned_vals}

# Plot 5 subplot (2 baris x 3 kolom, 1 slot kosong)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

x = np.arange(len(algo_names))
width = 0.35

for i, metric_name in enumerate(metrics.keys()):
    ax = axes[i]
    base_vals  = results[metric_name]['Baseline']
    tuned_vals = results[metric_name]['Tuning']

    bars1 = ax.bar(x - width/2, base_vals, width, label='Baseline', color='#1f77b4')
    bars2 = ax.bar(x + width/2, tuned_vals, width, label='Tuning', color='#ff7f0e')

    ax.set_title(metric_name)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names, rotation=30, ha='right')

    # label angka di atas tiap bar
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)

    if i == 0:
        ax.legend()

# Kosongkan subplot ke-6 (karena cuma 5 metrik)
fig.delaxes(axes[5])

plt.tight_layout()
plt.show()

# ======================================================================
# FEATURE IMPORTANCE DR KETIGA ALGORITMA
# ======================================================================

coef = lr.coef_[0]

feature_importance_lr = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': coef
})

feature_importance_lr['Absolute'] = feature_importance_lr['Coefficient'].abs()

#diurutinn pengaruh terbesar ke terkecil
feature_importance_lr = feature_importance_lr.sort_values(
    by='Absolute',
    ascending=False
)

plt.figure(figsize=(10,6))
plt.barh(
    feature_importance_lr['Feature'][:10],
    feature_importance_lr['Coefficient'][:10]
)

plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importance - Logistic Regression")
plt.xlabel("Coefficient")
plt.grid(axis='x', alpha=0.3)
plt.show()

importance_rf = rf.feature_importances_

feature_importance_rf = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importance_rf
})

feature_importance_rf = feature_importance_rf.sort_values(
    by='Importance',
    ascending=False
)

top10 = feature_importance_rf.head(10)

# Plot
plt.figure(figsize=(10,6))
bars = plt.barh(top10['Feature'], top10['Importance'])

plt.gca().invert_yaxis()

for bar in bars:
    plt.text(
        bar.get_width() + 0.002,
        bar.get_y() + bar.get_height()/2,
        f'{bar.get_width():.3f}',
        va='center',
        fontsize=9
    )

plt.title("Top 10 Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.grid(axis='x', alpha=0.3)
plt.show()

# Mengambil feature importance
importance_xgb = xgb.feature_importances_

feature_importance_xgb = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importance_xgb
})

# Urutkan dari terbesar
feature_importance_xgb = feature_importance_xgb.sort_values(
    by='Importance',
    ascending=False
)

top10 = feature_importance_xgb.head(10)
plt.figure(figsize=(10,6))
bars = plt.barh(top10['Feature'], top10['Importance'])

plt.gca().invert_yaxis()

for bar in bars:
    plt.text(
        bar.get_width() + 0.002,
        bar.get_y() + bar.get_height()/2,
        f'{bar.get_width():.3f}',
        va='center',
        fontsize=9
    )

plt.title("Top 10 Feature Importance - XGBoost")
plt.xlabel("Importance Score")
plt.grid(axis='x', alpha=0.3)
plt.show()

# ======================================================================
# ROC-AUC CURVE & PRECISON-RECALL CURVE
# ======================================================================

from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Logistic Regression
precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_prob_lr)
ap_lr = average_precision_score(y_test, y_prob_lr)

# Random Forest
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_prob_rf)
ap_rf = average_precision_score(y_test, y_prob_rf)

# XGBoost
precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)
ap_xgb = average_precision_score(y_test, y_prob_xgb)

plt.figure(figsize=(8,6))
plt.plot(
    recall_lr,
    precision_lr,
    label=f'Logistic Regression (AP = {ap_lr:.3f})',
    linewidth=2
)

plt.plot(
    recall_rf,
    precision_rf,
    label=f'Random Forest (AP = {ap_rf:.3f})',
    linewidth=2
)

plt.plot(
    recall_xgb,
    precision_xgb,
    label=f'XGBoost (AP = {ap_xgb:.3f})',
    linewidth=2
)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.legend(loc='lower left')
plt.grid(True)

plt.show()

# BASE MODEL

# Logistic Regression
precision_lr_base, recall_lr_base, _ = precision_recall_curve(
    y_test, y_prob_lr_base
)
ap_lr_base = average_precision_score(y_test, y_prob_lr_base)

# Random Forest
precision_rf_base, recall_rf_base, _ = precision_recall_curve(
    y_test, y_prob_rf_base
)
ap_rf_base = average_precision_score(y_test, y_prob_rf_base)

# XGBoost
precision_xgb_base, recall_xgb_base, _ = precision_recall_curve(
    y_test, y_prob_xgb_base
)
ap_xgb_base = average_precision_score(y_test, y_prob_xgb_base)

plt.figure(figsize=(8,6))

plt.plot(
    recall_lr_base,
    precision_lr_base,
    label=f'Logistic Regression (AP = {ap_lr_base:.3f})',
    linewidth=2
)

plt.plot(
    recall_rf_base,
    precision_rf_base,
    label=f'Random Forest (AP = {ap_rf_base:.3f})',
    linewidth=2
)

plt.plot(
    recall_xgb_base,
    precision_xgb_base,
    label=f'XGBoost (AP = {ap_xgb_base:.3f})',
    linewidth=2
)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison (Baseline)')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.grid(True)
plt.legend(loc='lower left')
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Dictionary model baseline
models = {
    "Logistic Regression": (lr_base, X_test_scaled),
    "Random Forest": (rf_base, X_test),
    "XGBoost": (xgb_base, X_test)
}

plt.figure(figsize=(8,6))

for name, (model, X) in models.items():

    y_prob = model.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f'{name} (AUC = {auc:.3f})'
    )

# Garis acuan
plt.plot([0,1],[0,1],'k--',label='Random Guess')

plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve Comparison (Baseline)')
plt.legend(loc='lower right')
plt.grid(True)

plt.show()

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Dictionary model
models = {
    "Logistic Regression": (lr, X_test_scaled),
    "Random Forest": (rf, X_test),
    "XGBoost": (xgb, X_test)
}

plt.figure(figsize=(8,6))

for name, (model, X) in models.items():

    # Prediksi
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f'{name} (AUC = {auc:.3f})'
    )

# Garis acak
plt.plot([0,1],[0,1],'k--',label='Random Guess')

plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

auc_lr = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])
auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
auc_xgb = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
models = ['Logistic Regression', 'Random Forest', 'XGBoost']

# Prediksi
predictions = [y_pred_lr, y_pred_rf, y_pred_xgb]

# Hitung metrik
accuracy = [accuracy_score(y_test, y) for y in predictions]
precision = [precision_score(y_test, y, zero_division=0) for y in predictions]
recall = [recall_score(y_test, y, zero_division=0) for y in predictions]
f1 = [f1_score(y_test, y, zero_division=0) for y in predictions]
auc = [auc_lr, auc_rf, auc_xgb]

# Plot
x = np.arange(len(models))
width = 0.15

plt.figure(figsize=(12,6))

bars1 = plt.bar(x-2*width, accuracy, width, label='Accuracy')
bars2 = plt.bar(x-width, precision, width, label='Precision')
bars3 = plt.bar(x, recall, width, label='Recall')
bars4 = plt.bar(x+width, f1, width, label='F1-Score')
bars5 = plt.bar(x+2*width, auc, width, label='AUC')

# Tambahkan nilai di atas setiap batang
for bars in [bars1, bars2, bars3, bars4, bars5]:
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{bar.get_height():.3f}',
            ha='center',
            va='bottom',
            fontsize=8
        )

plt.xticks(x, models)
plt.ylim(0, 1.15)
plt.ylabel('Score')
plt.title('Perbandingan Performa dari ke-3 model Logistic Regression, Random Forest, and XGBoost')
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.show()

# ======================================================================
# LEARNING CURVE KE-3 ALGORITMA
# ======================================================================

#BASELINE
fig, axes = plt.subplots(1, 3, figsize=(18,5))

models = [
    ("Logistic Regression", lr_base, X_train_scaled),
    ("Random Forest", rf_base, X_train),
    ("XGBoost", xgb_base, X_train)
]

for ax, (name, model, X_data) in zip(axes, models):

    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X_data,
        y=y_train,
        cv=5,
        scoring='accuracy',
        train_sizes=np.linspace(0.3, 1.0, 6),
        shuffle=True,
        random_state=42,
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    ax.plot(train_sizes, train_mean, marker='o', label='Training Accuracy')
    ax.plot(train_sizes, val_mean, marker='o', label='Validation Accuracy')

    ax.set_title(name)
    ax.set_xlabel("Training Size")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.65, 1.00)
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18,5))

models = [
    ("Logistic Regression", lr, X_train_scaled),
    ("Random Forest", rf, X_train),
    ("XGBoost", xgb, X_train)
]

for ax, (name, model, X_data) in zip(axes, models):

    train_sizes, train_scores, val_scores = learning_curve(
    estimator=model,
    X=X_data,
    y=y_train,
    cv=5,
    scoring='accuracy',
    train_sizes=np.linspace(0.3, 1.0, 6),
    shuffle=True,
    random_state=42,
    n_jobs=-1
)

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    ax.plot(train_sizes, train_mean, marker='o', label='Training Accuracy')
    ax.plot(train_sizes, val_mean, marker='o', label='Validation Accuracy')

    ax.set_title(name)
    ax.set_xlabel("Training Size")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.65, 1.00)  
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

# ======================================================================
# CROSS VALIDATION
# ======================================================================

#BASELINE

from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_summary = []
for name, model, X_data in [
    ("Logistic Regression", lr_base, X_train_scaled),
    ("Random Forest", rf_base, X_train),
    ("XGBoost", xgb_base, X_train)
]:
    scores = cross_val_score(model, X_data, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)
    cv_summary.append({'Model': name, 'ROC-AUC Mean': scores.mean(), 'ROC-AUC Std': scores.std()})
    print(f"{name:22s} | ROC-AUC CV: {scores.mean():.4f} (+/- {scores.std():.4f})")

cv_summary_df = pd.DataFrame(cv_summary)
cv_summary_df

from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_summary = []
for name, model, X_data in [
    ("Logistic Regression", lr, X_train_scaled),
    ("Random Forest", rf, X_train),
    ("XGBoost", xgb, X_train)
]:
    scores = cross_val_score(model, X_data, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)
    cv_summary.append({'Model': name, 'ROC-AUC Mean': scores.mean(), 'ROC-AUC Std': scores.std()})
    print(f"{name:22s} | ROC-AUC CV: {scores.mean():.4f} (+/- {scores.std():.4f})")

cv_summary_df = pd.DataFrame(cv_summary)
cv_summary_df

# ======================================================================
# VALIDATION CURVE
# ======================================================================

from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(18,5))

# ================= Logistic Regression =================
param_range_lr = [0.1, 0.5, 1, 5, 10]

train_scores, val_scores = validation_curve(
    estimator=lr_base,
    X=X_train_scaled,
    y=y_train,
    param_name="C",
    param_range=param_range_lr,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

axes[0].plot(param_range_lr, train_scores.mean(axis=1),
             marker='o', linewidth=2, label='Training')

axes[0].plot(param_range_lr, val_scores.mean(axis=1),
             marker='o', linewidth=2, label='Validation')

axes[0].set_xscale("log")
axes[0].set_title("Validation Curve Logistic Regression")
axes[0].set_xlabel("C")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

# ================= Random Forest =================
param_range_rf = [10, 15, 20, 25, 30]

train_scores, val_scores = validation_curve(
    estimator=rf_base,
    X=X_train,
    y=y_train,
    param_name="max_depth",
    param_range=param_range_rf,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

axes[1].plot(param_range_rf, train_scores.mean(axis=1),
             marker='o', linewidth=2, label='Training')

axes[1].plot(param_range_rf, val_scores.mean(axis=1),
             marker='o', linewidth=2, label='Validation')

axes[1].set_title("Validation Curve Random Forest")
axes[1].set_xlabel("max_depth")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(alpha=0.3)

# ================= XGBoost =================
param_range_xgb = [3, 4, 5, 6, 7, 8]

train_scores, val_scores = validation_curve(
    estimator=xgb_base,
    X=X_train,
    y=y_train,
    param_name="max_depth",
    param_range=param_range_xgb,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

axes[2].plot(param_range_xgb, train_scores.mean(axis=1),
             marker='o', linewidth=2, label='Training')

axes[2].plot(param_range_xgb, val_scores.mean(axis=1),
             marker='o', linewidth=2, label='Validation')

axes[2].set_title("Validation Curve XGBoost")
axes[2].set_xlabel("max_depth")
axes[2].set_ylabel("Accuracy")
axes[2].legend()
axes[2].grid(alpha=0.3)

for ax in axes:
    ax.set_ylim(0.65, 1.00)

plt.tight_layout()
plt.show()

from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(18,5))

# Logistic Regression (Parameter C)
param_range_lr = [0.1, 0.5, 1, 5, 10]

train_scores, val_scores = validation_curve(
    estimator=lr,
    X=X_train_scaled,
    y=y_train,
    param_name="C",
    param_range=param_range_lr,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

axes[0].plot(param_range_lr, train_scores.mean(axis=1),
             marker='o', linewidth=2, label='Training')

axes[0].plot(param_range_lr, val_scores.mean(axis=1),
             marker='o', linewidth=2, label='Validation')

axes[0].set_xscale("log")
axes[0].set_title("Validation Curve Logistic Regression")
axes[0].set_xlabel("C")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Random Forest (Parameter max_depth)
param_range_rf = [10, 15, 20, 25, 30]

train_scores, val_scores = validation_curve(
    estimator=rf,
    X=X_train,
    y=y_train,
    param_name="max_depth",
    param_range=param_range_rf,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

axes[1].plot(param_range_rf, train_scores.mean(axis=1),
             marker='o', linewidth=2, label='Training')

axes[1].plot(param_range_rf, val_scores.mean(axis=1),
             marker='o', linewidth=2, label='Validation')

axes[1].set_title("Validation Curve Random Forest")
axes[1].set_xlabel("max_depth")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(alpha=0.3)

# XGBoost (Parameter max_depth)
param_range_xgb = [3, 4, 5, 6, 7, 8]

train_scores, val_scores = validation_curve(
    estimator=xgb,
    X=X_train,
    y=y_train,
    param_name="max_depth",
    param_range=param_range_xgb,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

axes[2].plot(param_range_xgb, train_scores.mean(axis=1),
             marker='o', linewidth=2, label='Training')

axes[2].plot(param_range_xgb, val_scores.mean(axis=1),
             marker='o', linewidth=2, label='Validation')

axes[2].set_title("Validation Curve XGBoost")
axes[2].set_xlabel("max_depth")
axes[2].set_ylabel("Accuracy")
axes[2].legend()
axes[2].grid(alpha=0.3)

for ax in axes:
    ax.set_ylim(0.65, 1.00)

plt.tight_layout()
plt.show()

# ======================================================================
# # Data Balancing (SMOTE)
# ======================================================================

# ==========================================
# BASELINE + SMOTE (Logistic Regression)
# ==========================================

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

print("Distribusi kelas sebelum SMOTE:", dict(y_train.value_counts()))
print("Distribusi kelas setelah SMOTE :", dict(pd.Series(y_train_sm).value_counts()))

lr_balanced_base = LogisticRegression(
    random_state=42,
    max_iter=1000
)

lr_balanced_base.fit(X_train_sm, y_train_sm)

y_pred_lr_bal_base = lr_balanced_base.predict(X_test_scaled)

print("\nAccuracy Logistic Regression Baseline + SMOTE:",
      accuracy_score(y_test, y_pred_lr_bal_base))

print(classification_report(y_test, y_pred_lr_bal_base))

# ==========================================
# HYPERPARAMETER TUNING + SMOTE (Logistic Regression)
# ==========================================

lr_balanced_tuned = LogisticRegression(
    **grid_lr.best_params_,
    random_state=42,
    max_iter=1000
)

lr_balanced_tuned.fit(X_train_sm, y_train_sm)
y_pred_lr_bal_tuned = lr_balanced_tuned.predict(X_test_scaled)

print("\nAccuracy Logistic Regression Tuning + SMOTE:",
      accuracy_score(y_test, y_pred_lr_bal_tuned))

print(classification_report(y_test, y_pred_lr_bal_tuned))

# ==========================================
# BASELINE + SMOTE (XGBoost)
# ==========================================

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("Distribusi kelas sebelum SMOTE:", dict(y_train.value_counts()))
print("Distribusi kelas setelah SMOTE :", dict(pd.Series(y_train_sm).value_counts()))

xgb_balanced_base = XGBClassifier(
    random_state=42,
    eval_metric='logloss'
)

xgb_balanced_base.fit(X_train_sm, y_train_sm)

y_pred_xgb_bal_base = xgb_balanced_base.predict(X_test)

print("\nAccuracy XGBoost Baseline + SMOTE:",
      accuracy_score(y_test, y_pred_xgb_bal_base))

print(classification_report(y_test, y_pred_xgb_bal_base))

# ==========================================
# HYPERPARAMETER TUNING + SMOTE (XGBoost)
# ==========================================

xgb_balanced_tuned = XGBClassifier(**random_xgb.best_params_, random_state=42, eval_metric='logloss')
xgb_balanced_tuned.fit(X_train_sm, y_train_sm)

y_pred_xgb_bal_tuned = xgb_balanced_tuned.predict(X_test)
print("\nAccuracy XGBoost Tuning + SMOTE:", accuracy_score(y_test, y_pred_xgb_bal_tuned))
print(classification_report(y_test, y_pred_xgb_bal_tuned))

# ==========================================
# BASELINE + SMOTE (Random Forest)
# ==========================================

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("Distribusi kelas sebelum SMOTE:", dict(y_train.value_counts()))
print("Distribusi kelas setelah SMOTE :", dict(pd.Series(y_train_sm).value_counts()))

rf_balanced_base = RandomForestClassifier(
    random_state=42
)

rf_balanced_base.fit(X_train_sm, y_train_sm)

# Prediksi
y_pred_rf_bal_base = rf_balanced_base.predict(X_test)

print("\nAccuracy Random Forest Baseline + SMOTE:",
      accuracy_score(y_test, y_pred_rf_bal_base))

print(classification_report(y_test, y_pred_rf_bal_base))

# ==========================================
# HYPERPARAMETER TUNING + SMOTE (Random Forest)
# ==========================================

rf_balanced_tuned = RandomForestClassifier(
    **random_rf.best_params_,
    random_state=42
)

rf_balanced_tuned.fit(X_train_sm, y_train_sm)

# Prediksi
y_pred_rf_bal_tuned = rf_balanced_tuned.predict(X_test)

print("\nAccuracy Random Forest Tuning + SMOTE:",
      accuracy_score(y_test, y_pred_rf_bal_tuned))

print(classification_report(y_test, y_pred_rf_bal_tuned))

# ---------------- Confusion Matrix: BASELINE + SMOTE (ketiga algoritma) ----------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

labels = ["Tidak Terlambat", "Terlambat"]

# Logistic Regression + SMOTE (Baseline)
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_lr_bal_base,
    display_labels=labels,
    cmap="Blues",
    ax=axes[0]
)
axes[0].set_title("Baseline Logistic Regression + SMOTE")

# Random Forest + SMOTE (Baseline)
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_rf_bal_base,
    display_labels=labels,
    cmap="Blues",
    ax=axes[1]
)
axes[1].set_title("Baseline Random Forest + SMOTE")

# XGBoost + SMOTE (Baseline)
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_xgb_bal_base,
    display_labels=labels,
    cmap="Blues",
    ax=axes[2]
)
axes[2].set_title("Baseline XGBoost + SMOTE")

plt.tight_layout()
plt.show()

# ---------------- Confusion Matrix: HYPERPARAMETER TUNING + SMOTE (ketiga algoritma) ----------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

labels = ["Tidak Terlambat", "Terlambat"]

# lr + SMOTE (Tuning)
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_lr_bal_tuned,
    display_labels=labels,
    cmap="Blues",
    ax=axes[0]
)
axes[0].set_title("Logistic Regression (Tuning) + SMOTE")

# Random Forest + SMOTE (Tuning)
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_rf_bal_tuned,
    display_labels=labels,
    cmap="Blues",
    ax=axes[1]
)
axes[1].set_title("Random Forest (Tuning) + SMOTE")

# XGBoost + SMOTE (Tuning)
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_xgb_bal_tuned,
    display_labels=labels,
    cmap="Blues",
    ax=axes[2]
)
axes[2].set_title("XGBoost (Tuning) + SMOTE")

plt.tight_layout()
plt.show()

# ======================================================================
# PERBANDINGAN BASELINE VS HYPERPARAMETER TUNING TESTING&TRAINING
# ======================================================================

# ==== SMOTE resample (sekali aja, dipakai bertiga) ====
smote_plot = SMOTE(random_state=42)
X_train_sm_plot, y_train_sm_plot = smote_plot.fit_resample(X_train, y_train)

smote_plot_scaled = SMOTE(random_state=42)
X_train_sm_scaled_plot, y_train_sm_scaled_plot = smote_plot_scaled.fit_resample(X_train_scaled, y_train)

# ==== Fit ulang 4 kondisi x 3 algoritma, nama variabel baru biar aman ====

# --- Logistic Regression (pakai data scaled) ---
lr_plot_base = LogisticRegression(random_state=42, max_iter=1000)
lr_plot_base.fit(X_train_scaled, y_train)

lr_plot_base_sm = LogisticRegression(random_state=42, max_iter=1000)
lr_plot_base_sm.fit(X_train_sm_scaled_plot, y_train_sm_scaled_plot)

lr_plot_tuned = LogisticRegression(**grid_lr.best_params_, random_state=42, max_iter=1000)
lr_plot_tuned.fit(X_train_scaled, y_train)

lr_plot_tuned_sm = LogisticRegression(**grid_lr.best_params_, random_state=42, max_iter=1000)
lr_plot_tuned_sm.fit(X_train_sm_scaled_plot, y_train_sm_scaled_plot)

# --- Random Forest ---
rf_plot_base = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_plot_base.fit(X_train, y_train)

rf_plot_base_sm = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_plot_base_sm.fit(X_train_sm_plot, y_train_sm_plot)

rf_plot_tuned = RandomForestClassifier(**random_rf.best_params_, random_state=42, n_jobs=-1)
rf_plot_tuned.fit(X_train, y_train)

rf_plot_tuned_sm = RandomForestClassifier(**random_rf.best_params_, random_state=42, n_jobs=-1)
rf_plot_tuned_sm.fit(X_train_sm_plot, y_train_sm_plot)

# --- XGBoost ---
xgb_plot_base = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
xgb_plot_base.fit(X_train, y_train)

xgb_plot_base_sm = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
xgb_plot_base_sm.fit(X_train_sm_plot, y_train_sm_plot)

xgb_plot_tuned = XGBClassifier(**random_xgb.best_params_, random_state=42, eval_metric='logloss', n_jobs=-1)
xgb_plot_tuned.fit(X_train, y_train)

xgb_plot_tuned_sm = XGBClassifier(**random_xgb.best_params_, random_state=42, eval_metric='logloss', n_jobs=-1)
xgb_plot_tuned_sm.fit(X_train_sm_plot, y_train_sm_plot)

# ==== Kumpulkan semua hasil ====
algo_data = {
    'Logistic Regression': {
        'Baseline\nsebelum SMOTE': (lr_plot_base, X_train_scaled, y_train, X_test_scaled),
        'Baseline\nsesudah SMOTE': (lr_plot_base_sm, X_train_sm_scaled_plot, y_train_sm_scaled_plot, X_test_scaled),
        'Tuning\nsebelum SMOTE': (lr_plot_tuned, X_train_scaled, y_train, X_test_scaled),
        'Tuning\nsesudah SMOTE': (lr_plot_tuned_sm, X_train_sm_scaled_plot, y_train_sm_scaled_plot, X_test_scaled),
    },
    'Random Forest': {
        'Baseline\nsebelum SMOTE': (rf_plot_base, X_train, y_train, X_test),
        'Baseline\nsesudah SMOTE': (rf_plot_base_sm, X_train_sm_plot, y_train_sm_plot, X_test),
        'Tuning\nsebelum SMOTE': (rf_plot_tuned, X_train, y_train, X_test),
        'Tuning\nsesudah SMOTE': (rf_plot_tuned_sm, X_train_sm_plot, y_train_sm_plot, X_test),
    },
    'XGBoost': {
        'Baseline\nsebelum SMOTE': (xgb_plot_base, X_train, y_train, X_test),
        'Baseline\nsesudah SMOTE': (xgb_plot_base_sm, X_train_sm_plot, y_train_sm_plot, X_test),
        'Tuning\nsebelum SMOTE': (xgb_plot_tuned, X_train, y_train, X_test),
        'Tuning\nsesudah SMOTE': (xgb_plot_tuned_sm, X_train_sm_plot, y_train_sm_plot, X_test),
    },
}

# ==== Plot: 3 subplot berdampingan ====
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax, (algo_name, kondisi) in zip(axes, algo_data.items()):
    labels = list(kondisi.keys())
    train_acc, test_acc, gap = [], [], []

    for k in labels:
        model, Xtr, ytr, Xte = kondisi[k]
        tr = accuracy_score(ytr, model.predict(Xtr)) * 100
        te = accuracy_score(y_test, model.predict(Xte)) * 100
        train_acc.append(tr)
        test_acc.append(te)
        gap.append(tr - te)

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, train_acc, width, label='Training', color='#4C72B0')
    bars2 = ax.bar(x + width/2, test_acc, width, label='Testing', color='#DD8452')

    for bars in [bars1, bars2]:
        ax.bar_label(bars, fmt='%.1f%%', padding=2, fontsize=8)

    for i, g in enumerate(gap):
        ax.text(x[i], max(train_acc[i], test_acc[i]) + 5, f'Gap:\n{g:.2f}%',
                ha='center', fontsize=8, fontweight='bold',
                color='red' if abs(g) > 10 else 'green')

    ax.set_title(algo_name, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)
    if ax == axes[0]:
        ax.set_ylabel('Accuracy (%)')
    ax.legend(fontsize=8)

fig.suptitle('Perbandingan Training vs Testing Accuracy\nBaseline vs Tuning, Sebelum vs Sesudah SMOTE', fontsize=13)
plt.tight_layout()
plt.show()

# ======================================================================
# # Ensemble (Voting Classifier)
# ======================================================================

from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline

# ==========================================
# BASELINE (Voting: LR + RF + XGB)
# ==========================================

# Baseline Logistic Regression
lr_pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        random_state=42,
        max_iter=1000
    )
)

# Baseline Random Forest
rf_base = RandomForestClassifier(
    random_state=42
)

# Baseline XGBoost
xgb_base = XGBClassifier(
    random_state=42,
    eval_metric='logloss'
)

# Voting Classifier Baseline
voting_clf = VotingClassifier(
    estimators=[
        ('lr', lr_pipeline),
        ('rf', rf_base),
        ('xgb', xgb_base)
    ],
    voting='soft'
)

voting_clf.fit(X_train, y_train)

y_pred_voting = voting_clf.predict(X_test)
y_prob_voting = voting_clf.predict_proba(X_test)[:, 1]

print("Accuracy Voting Ensemble (Baseline):",
      accuracy_score(y_test, y_pred_voting))

print(classification_report(y_test, y_pred_voting))

# ==========================================
# HYPERPARAMETER TUNING (Voting: LR + RF + XGB)
# ==========================================

from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline

lr_pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(**grid_lr.best_params_, max_iter=1000, random_state=42)
)

voting_clf = VotingClassifier(
    estimators=[
        ('lr', lr_pipeline),
        ('rf', rf),
        ('xgb', xgb)
    ],
    voting='soft'
)

voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)
y_prob_voting = voting_clf.predict_proba(X_test)[:, 1]

print("Accuracy Voting Ensemble:", accuracy_score(y_test, y_pred_voting))
print(classification_report(y_test, y_pred_voting))

# ==========================================
# BASELINE (Voting: RF + XGB)
# ==========================================

# Baseline Random Forest
rf_base = RandomForestClassifier(
    random_state=42
)

# Baseline XGBoost
xgb_base = XGBClassifier(
    random_state=42,
    eval_metric='logloss'
)

# Voting Baseline RF + XGBoost
voting_rf_xgb = VotingClassifier(
    estimators=[
        ('rf', rf_base),
        ('xgb', xgb_base)
    ],
    voting='soft'
)

# Training
voting_rf_xgb.fit(X_train, y_train)

# Prediksi
y_pred_vote = voting_rf_xgb.predict(X_test)
y_prob_vote = voting_rf_xgb.predict_proba(X_test)[:, 1]

# Evaluasi
print("Accuracy Voting (RF + XGBoost) Baseline:",
      accuracy_score(y_test, y_pred_vote))

print(classification_report(y_test, y_pred_vote))

# ==========================================
# HYPERPARAMETER TUNING (Voting: RF + XGB)
# ==========================================

# Voting RF + XGBoost (Tuning)
voting_rf_xgb = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('xgb', xgb)
    ],
    voting='soft'
)

# Training
voting_rf_xgb.fit(X_train, y_train)

# Prediksi
y_pred_vote = voting_rf_xgb.predict(X_test)
y_prob_vote = voting_rf_xgb.predict_proba(X_test)[:, 1]

# Evaluasi
print("Accuracy Voting (RF + XGBoost):",
      accuracy_score(y_test, y_pred_vote))

print(classification_report(y_test, y_pred_vote))

# ======================================================================
# KESIMPULAN
# ======================================================================
 
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
 
model_names = ['Logistic Regression', 'Random Forest', 'XGBoost']
preds = [y_pred_lr, y_pred_rf, y_pred_xgb]
probs = [y_prob_lr, y_prob_rf, y_prob_xgb]
 
final_summary = pd.DataFrame([
    {
        'Model': name,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'F1-Score': f1_score(y_test, pred),
        'ROC-AUC': roc_auc_score(y_test, prob)
    }
    for name, pred, prob in zip(model_names, preds, probs)
]).round(4).sort_values('F1-Score', ascending=False).reset_index(drop=True)
 
print("\n=== FINAL SUMMARY: Perbandingan Semua Model (Hasil Tuning) ===")
print(final_summary)