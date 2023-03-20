# Imports

import pickle
import os
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
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
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_fp = os.path.join(DIRPATH, "assets", "ml", "ml_components.pkl")
req_fp = os.path.join(DIRPATH, "assets", "ml", "requirements.txt")
eda_report_fp = os.path.join(DIRPATH, "assets", "ml", "eda-report.html")

# import some data to play with
iris = datasets.load_iris(return_X_y=False, as_frame=True)

df = iris['frame']
target_col = 'target'
# pandas profiling
profile = ProfileReport(df, title="Dataset", html={
                        'style': {'full_width': True}})
profile.to_file(eda_report_fp)

# Dataset Splitting
# Please specify
to_ignore_cols = [
    "ID",  # ID
    "Id", "id",
    target_col
]


num_cols = list(set(df.select_dtypes('number')) - set(to_ignore_cols))
cat_cols = list(set(df.select_dtypes(exclude='number')) - set(to_ignore_cols))
print(f"\n[Info] The '{len(num_cols)}' numeric columns are : {num_cols}\nThe '{len(cat_cols)}' categorical columns are : {cat_cols}")

X, y = df.iloc[:, :-1], df.iloc[:, -1].values


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

model = RandomForestClassifier(random_state=10)

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
      target_names=iris['target_names']))

# ConfusionMatrixDisplay.from_predictions(
#     y_eval, y_eval_pred, display_labels=iris['target_names'])

# Exportation
print(
    f"\n[Info] Exportation.\n")
to_export = {
    "labels": iris['target_names'],
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
