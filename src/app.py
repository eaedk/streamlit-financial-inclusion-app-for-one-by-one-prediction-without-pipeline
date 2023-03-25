import streamlit as st
import pandas as pd
import os
import pickle


# PAGE CONFIG : Must be the first line after the importation section
st.set_page_config(
    page_title="Financial Inclusion App", page_icon="🏦", layout="centered")

# Setup variables and constants
DIRPATH = os.path.dirname(os.path.realpath(__file__))
tmp_df_fp = os.path.join(DIRPATH, "assets", "tmp", "history.csv")
ml_core_fp = os.path.join(DIRPATH, "assets", "ml", "ml_components.pkl")
init_df = pd.DataFrame(
    {'household_size': [], 'year': [], 'age_of_respondent': [], 'job_type': [], 'location_type': [], 'gender_of_respondent': [
    ], 'relationship_with_head': [], 'education_level': [], 'marital_status': [], 'cellphone_access': [], 'country': []}
)

# FUNCTIONS


@st.cache_resource()  # stop the hot-reload to the function just bellow
def load_ml_components(fp):
    "Load the ml component to re-use in app"
    with open(fp, "rb") as f:
        object = pickle.load(f)
    return object


def convert_df(df):
    "Convert a dataframe so that it will be downloadable"
    return df.to_csv(index=False).encode('utf-8')


def setup(fp):
    "Setup the required elements like files, models, global variables, etc"

    # history frame
    if not os.path.exists(fp):
        df_history = init_df.copy()
    else:
        df_history = pd.read_csv(fp)

    df_history.to_csv(fp, index=False)

    return df_history


# Setup execution
ml_components_dict = load_ml_components(fp=ml_core_fp)

labels = ml_components_dict['labels']
num_cols = ml_components_dict['num_cols']
cat_cols = ml_components_dict['cat_cols']
num_imputer = ml_components_dict['num_imputer']
cat_imputer = ml_components_dict['cat_imputer']
scaler = ml_components_dict['scaler']
encoder = ml_components_dict['encoder']
col_to_opts = {c: opts for (c, opts) in zip(cat_cols, encoder.categories_)}
print(f"Avalaible options for each categorical variable : {col_to_opts}")
model = ml_components_dict['model']

print(f"\n[Info] ML components loaded: {list(ml_components_dict.keys())}")
print(f"\n[Info] Predictable labels: {labels}")
idx_to_labels = {i: l for (i, l) in enumerate(labels)}
print(f"\n[Info] Indexes to labels: {idx_to_labels}")

try:
    df_history
except:
    df_history = setup(tmp_df_fp)

# APP Interface
st.image(
    "https://zindi-public-release.s3.eu-west-2.amazonaws.com/uploads/competition/image/19/header_df9adbda-2360-406d-9dc2-2342c054c795.png",
)
# style = "background-image: url("https: // zindi-public-release.s3.eu-west-2.amazonaws.com/uploads/competition/image/19/header_df9adbda-2360-406d-9dc2-2342c054c795.png"), url("");"
# Title
st.title("🏦 Financial Inclusion App")

st.write(
    f"This app shows a simple demo of a Streamlit app for. Financial inclusion remains one of the main obstacles to economic and human development in Africa. For example, across Kenya, Rwanda, Tanzania, and Uganda only 9.1 million adults (or 14% of adults) have access to or use a commercial bank account. The objective of this app that embeds a machine learning model is to predict which individuals are most likely to have or use a bank account.")

# Main page

# Form
form = st.form(key="information", clear_on_submit=True)
with form:

    cols = st.columns((1, 1, 1))
    cols_002 = st.columns((1, 1,))
    # petal_length = cols[0].slider("What's the petal length?: ", 0.0, 10.0, 1.0)

    # df_input = pd.DataFrame(
    #     {"petal length (cm)": [cols[0].slider("What's the petal length? :", 0.0, 10.0, 1.0)],
    #      "petal width (cm)": [cols[1].slider("What's the petal width? :", 0.0, 10.0, 1.0)],
    #      "sepal length (cm)": [cols[0].slider("What's the sepal length? :", 0.0, 10.0, 1.0)],
    #      "sepal width (cm)": [cols[1].slider("What's the sepal width? :", 0.0, 10.0, 1.0)], }
    # )
    df_input = pd.DataFrame(
        {'household_size': [cols[0].slider(label="Select the household size", min_value=1, max_value=40, value=3, step=1)],
         'year': [cols[1].slider(label="Select the year", min_value=2016, max_value=2018, value=2016, step=1)],
         'age_of_respondent': [cols[2].slider(label="Select the respondent age", min_value=0, max_value=120, value=30, step=1)],
         'job_type': [cols_002[1].selectbox('Select the job type', col_to_opts['job_type'])],
         'location_type': [cols[1].selectbox('Select the location type', col_to_opts['location_type'])],
         'gender_of_respondent': [cols[2].selectbox('Select the respondent gender', col_to_opts['gender_of_respondent'])],
         'relationship_with_head': [cols[0].selectbox('Select the relationship with head', col_to_opts['relationship_with_head'])],
         'education_level': [cols[1].selectbox('Select the education level', col_to_opts['education_level'])],
         'marital_status': [cols[2].selectbox('Select the marital status', col_to_opts['marital_status'])],
         'cellphone_access': [cols_002[0].radio('Cellphone access ?', col_to_opts['cellphone_access'], horizontal=True)],
         'country': [cols[0].selectbox('Select the country', col_to_opts['country'])]
         }
    )
    print(
        f"\n[Info] Input information as dataframe: \n{df_input.to_markdown()}\n")

    submitted = st.form_submit_button(label="Submit")

    if submitted:
        try:
            st.success("Thanks!")
            st.balloons()
            # # Prediction of just the labels...!
            # prediction_output = end2end_pipeline.predict(df_input)
            # print(
            #     f"[Info] Prediction output (of type '{type(prediction_output)}') from passed input: {prediction_output}")
            # df_input['pred_label'] = prediction_output
            # df_input['pred_label'] = df_input['pred_label'].replace(
            #     idx_to_labels)
            # Prediction of just the labels and confident scores...!

            X_cat = encoder.transform(cat_imputer.transform(
                df_input[cat_cols])) if len(cat_cols) > 0 else None

            X_num = scaler.transform(num_imputer.transform(
                df_input[num_cols])) if len(num_cols) > 0 else None

            X = pd.concat([X_num, X_cat], axis=1)

            prediction_output = model.predict_proba(X)
            print(
                f"[Info] Prediction output (of type '{type(prediction_output)}') from passed input: {prediction_output} of shape {prediction_output.shape}")
            predicted_idx = prediction_output.argmax(axis=-1)
            print(f"[Info] Predicted indexes: {predicted_idx}")
            df_input['pred_label'] = predicted_idx
            predicted_labels = df_input['pred_label'].replace(idx_to_labels)
            df_input['pred_label'] = predicted_labels
            predicted_score = prediction_output[:, predicted_idx]
            df_input['confidence_score'] = predicted_score
            df_history = pd.concat([df_history, df_input],
                                   ignore_index=True).convert_dtypes()
            df_history.to_csv(tmp_df_fp, index=False)

        except Exception as e:
            st.error(
                "Oops something went wrong, contact the client service or the admin!")
            print(
                f"\n[Error] {e} \n")


# Expander
expander = st.expander("Check the history")
with expander:

    if submitted:
        st.dataframe(df_history)
        st.download_button(
            "Download this table as CSV",
            convert_df(df_history),
            "prediction_history.csv",
            "text/csv",
            key='download-csv'
        )
