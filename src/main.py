from src.data_load import load_data
from src.model_train import train_all
from src.model_evaluate import evaluate_all

def main():
    X_train, X_test, Y_train, Y_test = load_data()
    train_all(X_train, Y_train)
    models = ["linear_regression", "xgboost", "random_forest"]
    for model_name in models:
        metrics = evaluate_all(model_name, X_test, Y_test)
        print(f"{model_name} evaluation:", metrics)

if __name__ == "__main__":
    main()
