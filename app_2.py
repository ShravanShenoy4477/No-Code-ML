from operator import index
import streamlit as st
import plotly.express as px

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

with open('config.yaml') as file:
    config = yaml.safe_load(file)
authenticator = stauth.Authenticate(config['credentials'],
                                    config['cookie']['name'],
                                    config['cookie']['key'],
                                    config['cookie']['expiry_days'],
                                    config['preauthorized']
                                    )

placeholder = st.empty()

if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)
