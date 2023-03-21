# Imports

import pickle
import os
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn import datasets
from subprocess import call

# PATHS
DIRPATH = os.path.dirname(os.path.relpath(__file__))
ASSETS_DIR = os.path.join(DIRPATH, "assets",)
DATASET_DIR = os.path.join(ASSETS_DIR, "dataset")
SPECIFIC_DATASET_DIR = os.path.join(
    ASSETS_DIR, "dataset", "financial-inclusion-in-africa")
ml_fp = os.path.join(ASSETS_DIR, "ml", "ml_components.pkl")
req_fp = os.path.join(ASSETS_DIR, "ml", "requirements.txt")
eda_report_fp = os.path.join(ASSETS_DIR, "ml", "eda-report.html")

# Download dataset
print(
    f"\n[Info] Download and preparing dataset. \n")
call(
    f"gdown 1BbAgKLqnBM7C2_BU9nzPieN0b2dh0XLm  -O '{DATASET_DIR}/' ", shell=True)

call(
    f"unzip -o '{os.path.join(DATASET_DIR, 'financial-inclusion-in-africa.zip')}' -d '{SPECIFIC_DATASET_DIR}/' ", shell=True)

# import some data to play with

train = pd.read_csv(os.path.join(SPECIFIC_DATASET_DIR, 'Train.csv'))
test = pd.read_csv(os.path.join(SPECIFIC_DATASET_DIR, 'Test.csv'))
ss = pd.read_csv(os.path.join(SPECIFIC_DATASET_DIR, 'SampleSubmission.csv'))
print(
    f"[Info] Dataset loaded : shape={train.shape}\n{train.head().to_markdown()}")

df = train
target_col = 'bank_account'
# 1 indicates that the individual does have a bank account and 0 indicates that they do not.
target_names = ["no_account", "has_account"]


# pandas profiling
profile = ProfileReport(df, title="Dataset", html={
                        'style': {'full_width': True}})
profile.to_file(eda_report_fp)

# Dataset Splitting
# Please specify
to_ignore_cols = [
    "ID",  # ID
    "Id", "uniqueid",
    target_col
]


num_cols = list(set(df.select_dtypes('number')) - set(to_ignore_cols))
cat_cols = list(set(df.select_dtypes(exclude='number')) - set(to_ignore_cols))
print(f"\n[Info] The '{len(num_cols)}' numeric columns are : {num_cols}\nThe '{len(cat_cols)}' categorical columns are : {cat_cols}")

X, y = df[num_cols+cat_cols], df[target_col].values


X_train, X_eval, y_train, y_eval = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)

print(
    f"\n[Info] Dataset splitted : (X_train , y_train) = {(X_train.shape , y_train.shape)}, (X_eval y_eval) = {(X_eval.shape , y_eval.shape)}. \n")

y_train

# Modeling

# Imputers
num_imputer = SimpleImputer(strategy="mean").set_output(transform="pandas")
cat_imputer = SimpleImputer(
    strategy="most_frequent").set_output(transform="pandas")

# Scaler & Encoder
cat_ = 'auto'
if len(cat_cols) > 0:
    df_imputed_stacked_cat = cat_imputer.fit_transform(
        df
        .append(df)
        .append(df)
        [cat_cols])
    cat_ = OneHotEncoder(sparse=False, drop="first").fit(
        df_imputed_stacked_cat).categories_

encoder = OneHotEncoder(categories=cat_, sparse=False,
                        drop="first").set_output(transform="pandas")
scaler = StandardScaler().set_output(transform="pandas")

X_train_cat, X_train_num = None, None

if len(cat_cols) > 0:
    X_train_cat = encoder.fit_transform(
        cat_imputer.fit_transform(X_train[cat_cols]))

if len(num_cols) > 0:
    X_train_num = scaler.fit_transform(
        num_imputer.fit_transform(X_train[num_cols]))

X_train_ok = pd.concat([X_train_num, X_train_cat], axis=1)

# RandomForestClassifier(random_state=10)
model = AdaBoostClassifier(random_state=10)

# Training
print(
    f"\n[Info] Training.\n[Info] X_train : columns( {X_train.columns.tolist()}), shape: {X_train.shape} .\n")

model.fit(X_train_ok, y_train)

# Evaluation
print(
    f"\n[Info] Evaluation.\n")

X_eval_cat = encoder.transform(
    cat_imputer.transform(X_eval[cat_cols])) if len(cat_cols) > 0 else None

X_eval_num = scaler.transform(
    num_imputer.transform(X_eval[num_cols]))if len(num_cols) > 0 else None

X_eval_ok = pd.concat([X_eval_num, X_eval_cat], axis=1)

y_eval_pred = model.predict(X_eval_ok)

print(classification_report(y_eval, y_eval_pred,
      target_names=target_names))

# ConfusionMatrixDisplay.from_predictions(
#     y_eval, y_eval_pred, display_labels=target_names)

# Exportation
print(
    f"\n[Info] Exportation.\n")
to_export = {
    "labels": target_names,
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "num_imputer": num_imputer,
    "cat_imputer": cat_imputer,
    "scaler": scaler,
    "encoder": encoder,
    "model": model,
}


# save components to file
with open(ml_fp, 'wb') as file:
    pickle.dump(to_export, file)

# Requirements
# ! pip freeze > requirements.txt
call(f"pip freeze > {req_fp}", shell=True)

print(f"[Info] Dictionary to use to as base for dataframe filling :\n",{col: [] for col in X_train.columns})