import mlflow 

def calculate_sum(a, b):
    return a + b    


if __name__ == "__main__":
    #starting the srever of mlflow
    with mlflow.start_run():
        x,y=10,20
        sum = calculate_sum(x,y)

        #tracking the experiment with the mlflow
        mlflow.log_param("x", x)
        mlflow.log_param("y", y)
        mlflow.log_metric("sum", sum)
    