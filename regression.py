# regression.py (with hyperparameter tuning)
from utils import load_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_and_tune():
    df = load_data()
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models_and_params = {
        "Ridge": (Ridge(), {
            "alpha": [0.1, 1.0, 10.0]
        }),
        "Lasso": (Lasso(), {
            "alpha": [0.001, 0.01, 0.1]
        }),
        "RandomForest": (RandomForestRegressor(random_state=42), {
            "n_estimators": [50, 100],
            "max_depth": [None, 10, 20]
        })
    }

    for name, (model, params) in models_and_params.items():
        print(f"\nTuning {name}...")
        grid = GridSearchCV(model, params, cv=5, scoring='r2')
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        preds = best_model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        print(f"Best Params for {name}: {grid.best_params_}")
        print(f"{name}: MSE = {mse:.2f}, R2 = {r2:.2f}")

if __name__ == "__main__":
    train_and_tune()
