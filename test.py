from operator import index
import streamlit as st
import plotly.express as px
from pycaret.classification import setup, compare_models, pull, save_model, load_model, plot_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os


cat_cols = []


if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoNickML")
    choice = st.radio(
        "Navigation", ["Upload", "Profiling", "Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling":
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    choice_metric = st.selectbox(
        'Choose metric of choice', ['Recall', 'Accuracy', 'AUC', 'Prec.', 'F1', 'Kappa'])
    select_categorical_columns = st.multiselect(
        'Select categorical (nominal) values', df.columns)
    for column in select_categorical_columns:
        cat_cols.append(column)
    if st.button('Run Modelling'):
        setup(data=df, target=chosen_target, silent=True,
              categorical_features=cat_cols, )
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models(sort=choice_metric)
        compare_df = pull()
        st.dataframe(compare_df)
        # plot_model(best_model, 'auc')
        # plot_model(best_model, 'pr')
        plot_model(best_model, 'confusion_matrix')
        # plot_model(best_model, 'feature')
        save_model(best_model, 'best_model')

if choice == "Download":
    with open('best_model.pkl', 'rb') as f:
        st.download_button('Download Model', f, file_name="best_model.pkl")
