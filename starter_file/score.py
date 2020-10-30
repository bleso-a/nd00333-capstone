import json
import numpy as np
import os
from sklearn.externals import joblib


def init():
    global model_path
    model_path = Model.get_model_path(model_name='my-model')


def clean_data(data):

    x_df = ds.to_pandas_dataframe().dropna()
    x_df['Loan_Status'].replace(True, 1, inplace=True)
    x_df['Loan_Status'].replace(False, 0, inplace=True)
    x_df.Gender = x_df.Gender.map({'Male': 1, 'Female': 0})
    x_df.Married = x_df.Married.map({True: 1, False: 0})
    x_df.Dependents = x_df.Dependents.map({'0': 0, '1': 1, '2': 2, '3+': 3})
    x_df.Education = x_df.Education.map({'Graduate': 1, 'Not Graduate': 0})
    x_df.Self_Employed = x_df.Self_Employed.map({True: 1, False: 0})
    x_df.Property_Area = x_df.Property_Area.map(
        {'Urban': 2, 'Rural': 0, 'Semiurban': 1})
    x_df.drop('Loan_ID', axis=1, inplace=True)

    Xt = x_df.iloc[1:542, 1:11]
    yt = x_df.iloc[1:542, 11]

    return Xt, yt


def run(data):
    try:
        data = np.array(json.loads(data))
        result = model.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
