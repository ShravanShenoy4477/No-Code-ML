from operator import index
import streamlit as st
import plotly.express as px
import extra_streamlit_components as stx
from pycaret.classification import setup, compare_models, pull, save_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os
import streamlit_authenticator as stauth
import yaml
from yaml import safe_load
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from streamlit_tags import st_tags
from sklearn.model_selection import train_test_split
import time

columns = []
cat_per_column = []
col_cats_dict = {}
cat_cols = []

placeholder = st.empty()

norm = False
with open('config.yaml') as file:
    config = yaml.safe_load(file)
authenticator = stauth.Authenticate(config['credentials'],
                                    config['cookie']['name'],
                                    config['cookie']['key'],
                                    config['cookie']['expiry_days'],
                                    config['preauthorized']
                                    )

if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)
st.title("No Code Machine Learning Application")
# st.markdown(
#     """
#         <style>
# @font-face {
#   font-family: 'Times New Roman';
#   font-style: normal;
#   font-weight: 400;
#   src: url(https://fonts.gstatic.com/s/tangerine/v12/IurY6Y5j_oScZZow4VOxCZZM.woff2) format('woff2');
#   unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+2074, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD;
# }
#     html, body, [class*="css"]  {
#     font-family: 'Arial';
#     font-size: 18px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Software Engineering Lab Project")
    choice = st.radio(
        "Navigation", ["Introduction", "Authentication", "Upload", "Profiling", "Pre-Processing", "Modelling", "Download"], index=0)
    st.info("This project application helps you build and explore your data.")

if choice == "Introduction":
    st.write("No Code Machine Learning essentially means that the user has to apply negligible amount of code in a certain programming language to perform modelling, training and validating models. With this application, you can easily upload your dataset, profile it to gain insights, preprocess the data to ensure its suitability, and select the best model based on a metric of your choice.")
    st.write("This is targeted at very new users and people who are new to the field of Machine Learning. Learning multiple libraries, familiarizing yourself with the environment and coding steps to train and test your model can be daunting to people at first albeit pretty simple once you get a hang of it. Your first project would most often than not be a Kaggle competition or a challenge or just the playground fun tasks that they have that enable you to get high accuracy scores with relatively fewer steps. ")
    st.write("Happy Training")
    st.image("/Users/shravanshenoy/Desktop/NO_CODE_ML.png")

if choice == 'Authentication':
    login, signup = False, False

    authenticated = False

    if 'login' not in st.session_state:
        st.session_state.login = False
    if 'signup' not in st.session_state:
        st.session_state.signup = False
    st.session_state.login_succ = False
    st.session_state.signup_succ = False
    login = st.button("Login")
    signup = st.button("Sign Up")
    if login:
        st.session_state.login = True
        st.session_state.signup = False
    if signup:
        st.session_state.signup = True
        st.session_state.login = False

    if st.session_state.login:
        name, authentication_status, username = authenticator.login(
            'Login', 'main')
        if authentication_status:
            authenticated = True
            st.session_state.login_succ = True
        elif authentication_status == False:
            st.error('Username/password is incorrect')
            st.session_state.login_succ = False
        elif authentication_status == None:
            st.warning('Please enter your username and password')
            st.session_state.login_succ = False

    elif st.session_state.signup:
        try:
            if authenticator.register_user(
                    'Register user', preauthorization=False):
                st.success('User registered successfully')
                authenticated = True
                name = st.session_state["name"]
                st.session_state.signup_succ = True
            # placeholder.empty()
        except Exception as e:
            st.error(e)
            st.session_state.signup_succ = False

        with open('config.yaml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False)

    if authenticated:
        placeholder.empty()
        st.write(f'Welcome *{name}*')
        *_, col5, col6 = st.columns(6)
        with col6:
            authenticator.logout('Logout', 'main')

if choice == "Upload":
    if (st.session_state.login_succ or st.session_state.signup_succ):
        st.title("Uploading Your Dataset")
        st.empty()
        file = st.file_uploader(
            "Upload Your Dataset in JSON/CSV/Txt/XLS/Archive formats")
        if file:
            type_of_file = file.name.split('.')[-1]
            file_name = file.name.split('.')[0]
            if type_of_file == 'csv' or type_of_file == 'txt':
                sep = st.text_input('Enter separator', value=',')
                df = pd.read_csv(file, index_col=None, sep=sep)
            elif type_of_file == 'xlsx' or type_of_file == 'xls':
                df = pd.read_excel(file, index_col=None, header=1)
            elif type_of_file == 'json':
                df = pd.read_json(file)
            elif type_of_file == 'zip':
                import zipfile
                with zipfile.ZipFile(file) as z:
                    file_name = file_name + '.csv'
                    with z.open(file_name) as f:
                        df = pd.read_csv(f)
            st.dataframe(df)
            st.text('Progress Bar')
            my_bar = st.progress(34, text='Uploading Complete')
    else:
        st.text("Authentication not complete")

if choice == "Profiling":
    if (st.session_state.login_succ or st.session_state.signup_succ):
        time_before = time.time()
        st.title("Exploratory Data Analysis")
        profile_df = df.profile_report()
        profile_df.to_file("ProfileReport.json")
        time_after = time.time()
        duration = time_after-time_before
        st_profile_report(profile_df)
        print('Profiling time')
        print(duration)
        st.text('Progress Bar')
        my_bar = st.progress(51, text='Profiling Complete')
    else:
        st.text("Authentication not complete")

if choice == "Pre-Processing":
    if (st.session_state.login_succ or st.session_state.signup_succ):
        st.title("Pre-processing your data")
        st.session_state.preprocess = True

        if st.session_state.preprocess:
            # Drop Rows that have only NaN values
            df.dropna(axis=0, thresh=1, inplace=True)
            df.reset_index(inplace=True)
            df.drop(['index'], axis=1, inplace=True)
            df.fillna('N/A')
            st.text(
                'You should choose to split your dataset into Train-Validation-Test.')
            frac_input = st.text_input('Enter fraction of Train', value=0.8)
            frac_input = float(frac_input)
            # Drop Columns based on user Input
            choice_3 = st.radio(
                "Do you wish to delete some columns?", ['Yes', 'No'])
            if choice_3 == 'Yes':
                selected_columns = st.multiselect(
                    'Choose columns to remove', df.columns, key='multiselect_remove')
                st.write(f'You have selected column(s): {selected_columns}')
                df.drop(selected_columns, axis=1, inplace=True)
                st.text("Dataset after updation")
                st.dataframe(df)
            choice_4 = st.selectbox('Choose strategy to replace NaN values',
                                    ['None', 'mean', 'most_frequent', 'median'], index=0)
            selected_columns = st.multiselect(
                'Choose columns to modify', df.columns, key='multiselect_modify')

            # Replace Certain Missing Values
            for column in selected_columns:
                index_col = df.columns.get_loc(column)
                if choice_4 == 'mean':
                    # print("In mean")
                    imputer = SimpleImputer(
                        missing_values=np.nan, strategy='mean')
                elif choice_4 == 'most_frequent':
                    # print("In mf")
                    imputer = SimpleImputer(
                        missing_values=np.nan, strategy='most_frequent')
                elif choice_4 == 'median':
                    # print("In median")
                    imputer = SimpleImputer(
                        missing_values=np.nan, strategy='median')

                values = imputer.fit_transform(
                    df.iloc[:, index_col].values.reshape(-1, 1))
                df.iloc[:, index_col] = values.squeeze()
            df.to_csv('dataset.csv', index=None)
            st.text("Dataset after updation")
            st.dataframe(df)
            st.text('To perform standardization of numerical data')
            selected_column = st.selectbox(
                'Choose columns to modify', df.columns, key='select_scale', )
            choice_5 = st.selectbox('Choose strategy to feature scale',
                                    ['None', 'Standard', 'MinMax', 'Robust'], index=0)
            if choice_5 == 'Standard':
                scaler = StandardScaler()
                df[selected_column] = scaler.fit_transform(
                    df[selected_column].values.reshape(-1, 1))

            elif choice_5 == 'MinMax':
                scaler = MinMaxScaler()
                df[selected_column] = scaler.fit_transform(
                    df[selected_column].values.reshape(-1, 1))

            elif choice_5 == 'Robust':
                scaler = RobustScaler()
                df[selected_column] = scaler.fit_transform(
                    df[selected_column].values.reshape(-1, 1))

            elif choice_5 == 'None':
                pass

            df.to_csv('dataset.csv', index=None)
            st.text("Dataset after updation")
            st.dataframe(df)
            train_df = df.sample(frac=frac_input, random_state=25)
            test_df = df.drop(train_df.index)
            train_df.to_csv('dataset_train.csv', index=None)
            test_df.to_csv('dataset_test.csv', index=None)
            choice_yes = st.button('Do you wish to profile data again?')
            if choice_yes:
                profile_df = train_df.profile_report()
                profile_df.to_file("ProfileReport.json")
                st_profile_report(profile_df)
                st.text('Progress Bar')
                my_bar = st.progress(68, text='Pre-processing Complete')
    else:
        st.text("Authentication not complete")

if choice == "Modelling":
    if (st.session_state.login_succ or st.session_state.signup_succ):
        if 'preprocess' in st.session_state:
            if os.path.exists('./dataset_train.csv'):
                train_df = pd.read_csv('dataset_train.csv', index_col=None)
            if os.path.exists('./dataset_test.csv'):
                test_df = pd.read_csv('dataset_test.csv', index_col=None)
            choice_corr = st.radio('Choose type of task to perform',
                                   ['Classification', 'Regression'])
            chosen_target = st.selectbox(
                'Choose the Target Column', df.columns)
            if choice_corr == 'Classification':
                choice_metric = st.selectbox(
                    'Choose metric of choice', ['Recall', 'Accuracy', 'AUC', 'Prec.', 'F1', 'Kappa'])
            if choice_corr == 'Regression':
                choice_metric = st.selectbox(
                    'Choose metric of choice', ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE'])
            select_categorical_columns = st.multiselect(
                'Select categorical (nominal) values', df.columns)
            if select_categorical_columns.count != 0:
                for column in select_categorical_columns:
                    cat_cols.append(column)
            selected_ord_columns = st.multiselect(
                'Select categorical (ordinal) values', df.columns)
            if selected_ord_columns.count != 0:
                for column in selected_ord_columns:
                    columns.append(column)
                    keywords = st_tags(
                        label=f'Enter categories for {column}',
                        text='Press enter to add more',
                        value=[],
                        suggestions=[],
                        maxtags=10,
                        key=column)
                    col_cats_dict.update({column: keywords})
            select_fold = st.slider(
                'Choose the number of folds of cross-validation', 2, 10, 5)

            if st.button('Run Modelling'):
                if choice_corr == 'Regression':
                    time_before = time.time()
                    from pycaret.regression import *
                    setup(data=train_df, target=chosen_target, train_size=1.0, preprocess=True,
                          silent=True, test_data=test_df, ordinal_features=col_cats_dict, categorical_features=cat_cols, fold=select_fold, verbose=True)
                    setup_df = pull()
                    st.dataframe(setup_df)
                    best_model = compare_models(sort=choice_metric)
                    evaluate_model(best_model)
                    compare_df = pull()
                    st.dataframe(compare_df)
                    plot_model(best_model, plot='residuals')
                    plot_model(best_model, plot='feature')
                    st.text('These are predictions on test dataset')
                    predictions = predict_model(best_model)
                    st.dataframe(predictions)
                    time_after = time.time()
                    duration = time_after - time_before
                    print(duration)
                    save_model(best_model, 'best_model')

                if choice_corr == 'Classification':
                    time_before = time.time()
                    from pycaret.classification import *
                    setup(data=train_df, target=chosen_target, train_size=1.0, preprocess=True,
                          silent=True, test_data=test_df, ordinal_features=col_cats_dict, categorical_features=cat_cols, fold=select_fold, verbose=False)
                    setup_df = pull()
                    st.dataframe(setup_df)
                    best_model = compare_models(sort=choice_metric)
                    evaluate_model(best_model)
                    compare_df = pull()
                    st.dataframe(compare_df)
                    col1, col2 = st.columns(2)
                    with col1:
                        plot_model(best_model, 'auc',
                                   display_format='streamlit')
                        plot_model(best_model, 'pr',
                                   display_format='streamlit')
                    with col2:
                        plot_model(best_model, 'confusion_matrix',
                                   display_format='streamlit')
                        plot_model(best_model, 'feature',
                                   display_format='streamlit')
                    st.text('These are predictions on test dataset')
                    predictions = predict_model(best_model)
                    st.dataframe(predictions)
                    time_after = time.time()
                    duration = time_after - time_before
                    print('Modelling time')
                    print(duration)
                    save_model(best_model, 'best_model')
                st.text('Progress Bar')
                my_bar = st.progress(85, text='Modelling Complete')
    else:
        st.text("Authentication not complete")

if choice == "Download":
    if (st.session_state.login_succ or st.session_state.signup_succ):
        with open('best_model.pkl', 'rb') as f:
            choice_down = st.download_button('Download Model', f,
                                             file_name="best_model.pkl")
            if choice_down:
                st.text('Progress Bar')
                my_bar = st.progress(100, text='Download Complete')
    else:
        st.text("Authentication not complete")
