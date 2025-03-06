import pandas as pd
import numpy as np 
import os 
import argparse
import mlflow 
import mlflow.sklearn 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

def get_data():
    URL="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(URL, sep=";")
    return df 

def evaluate(y_true, y_pred): 
   mae= mean_absolute_error(y_true, y_pred) 
   mse=mean_squared_error(y_true, y_pred)
   rmse=np.sqrt(mean_squared_error(y_true, y_pred))
   r2=r2_score(y_true, y_pred) 
   return mae, mse, rmse, r2 


def main(n_estimators, max_depth): 
    #get data
    df=get_data()
    print(df)
    train, test= train_test_split(df)

    #train, test split
    X_train = train.drop(["quality"], axis=1)
    X_test = test.drop(["quality"], axis=1)

    y_train = train["quality"]
    y_test = test["quality"]

    #train the model
    lr=ElasticNet()
    lr.fit(X_train, y_train)
    pred= lr.predict(X_test)    

    rf=RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth) 
    rf.fit(X_train, y_train)
    pred= rf.predict(X_test)

    #evaluate the model
    mae, mse, rmse, r2=evaluate(y_test, pred)
    

    print(f"mean absolute error: {mae}, mean squared error: {mse}, root mean squared error: {rmse}, r2 score: {r2}")
   
if __name__ == "__main__":
    args= argparse.ArgumentParser()
    args.add_argument("--n_estimators", "-n", default=50, type=int)
    args.add_argument("--max_depth", "-m", default=5, type=int)
    parsed_args= args.parse_args() 
    try:
        main(n_estimators=parsed_args.n_estimators, max_depth=parsed_args.max_depth)
    except Exception as e:
        raise e
