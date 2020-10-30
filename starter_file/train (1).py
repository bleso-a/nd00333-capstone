from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory


url = 'https://cap.blob.core.windows.net/cap/train.csv'

ds = TabularDatasetFactory.from_delimited_files(url)


run = Run.get_context()


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
    yt = x_df[['Loan_Status']]

    return Xt, yt


x, y = clean_data(ds)



def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0,
                        help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100,
                        help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(
        C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')


if __name__ == '__main__':
    main()
