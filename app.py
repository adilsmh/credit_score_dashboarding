# --- LIBRARIES --- #
from http import client
from unicodedata import name
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# from ./scripts/

# --- APP TITLE --- #
st.title("NoLoanPred V2")

# --- DATA FETCHING --- #
data = pd.read_csv('./data/csv/train_test/application_test.csv',
                   sep=None, delim_whitespace=None, engine="python")
# --- DATA VISUALIZATION --- #
client_index = st.slider("Depeche toi à choisir ton client gros con va!!",
                      min_value=data.first_valid_index(), max_value=data.last_valid_index())
pert_cols = ["SK_ID_CURR", "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]

df = data[pert_cols]
client_df = df.iloc[client_index]

cat_col_names = df.select_dtypes(include=['object']).columns
num_col_names = df.drop(cat_col_names, axis=1)

# grid_cols = st.columns(3)
# for col in zip(grid_cols, df[cat_col_names].columns):
#     col = 

# for col in cat_col_names:
#     fig, ax = plt.figure()  
#     ax.set(ylabel="Categories",
#         xlabel="Number of clients",
#         title= f"Categorical distribution for \n {col}")
#     st.pyplot(sns.countplot(y=df[col], data=df))
    # st.bar_chart(df[col])


# --- PREDICTION --- #
model = joblib.load("./models/logistic_model.pkl")
X = data[["DAYS_BIRTH", "REGION_RATING_CLIENT_W_CITY",
       "REGION_RATING_CLIENT", "DAYS_LAST_PHONE_CHANGE", "DAYS_ID_PUBLISH",
       "REG_CITY_NOT_WORK_CITY", "FLAG_EMP_PHONE", "REG_CITY_NOT_LIVE_CITY",
       "FLAG_DOCUMENT_3"]].iloc[client_index]

# X = X.dropna()

# """Select features with "days" named headers to convert them into years"""
# for col in X.columns:
#     if "days" in col.lower():
#         X[col] = X[col].abs()
#         X[col] = X[col].div(365)
#         X[col] = X[col].round(decimals=0)
#         X[col] = pd.to_numeric(X[col], downcast="integer")

# if st.button("On verra si t'auras ton credit, mais ne rêves pas trop mon ptit loulou"):
#     prediction = model.predict(X)
#     prediction_proba = model.predict_proba(X)
#     st.text(prediction)
#     st.text(prediction_proba)



