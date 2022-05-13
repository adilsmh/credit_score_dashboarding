import os
import sys

# visualisation
import numpy as np
import pandas as pd

# preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

# --- DATA PREPROCESSING --- #


def preprocessing(data: pd.DataFrame, n_corr_fts=False):
    """
    Function for data preprocessing

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe containing data variables

    Returns
    -------
    X_train: np.array
    y_train: np.array
    X_test: np.array
    y_test: np.array
    """
    df = data.copy()  # define dataset copy

    corr_fts = data.corr()[["TARGET"]].sort_values("TARGET", ascending=False).head(
        10).index  # retrieve nine most target correlated features
    df = df[corr_fts]

    df = df.dropna()
    df.drop(["FLAG_DOCUMENT_3", "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_LIVE_CITY"],
            axis=1, inplace=True)  # drop co-correlated features

    """Select features with "days" named headers to convert them into years"""
    for col in df.columns:
        if "days" in col.lower():
            df[col] = df[col].abs()
            df[col] = df[col].div(365)
            df[col] = df[col].round(decimals=0)
            df[col] = pd.to_numeric(df[col], downcast="integer")
     
    X = df.drop("TARGET", axis=1)
    y = df["TARGET"]

    # define categorical features name list
    cat_col_names = list(
        ["REGION_RATING_CLIENT", "REG_CITY_NOT_WORK_CITY", "FLAG_EMP_PHONE"])
    # define discrete numerical features name list
    num_col_names = list(X.drop(cat_col_names, axis=1).columns)

    # standard scale numerical features
    X = StandardScaler().fit_transform(X[num_col_names])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True)  # train / test split
    # Over-sampling on minority class because of unbalancing
    X_train, y_train = SMOTE().fit_resample(X_train, y_train)

    return X_train, y_train, X_test, y_test
