import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from src.data_processing import build_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def load_data():
    """Load and preprocess data"""
    pipeline = build_pipeline()
    raw_data = pd.read_csv("data/raw/data.csv")
    return pipeline.fit_transform(raw_data)

def train_model():
    # Initialize MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Local MLflow server
    mlflow.set_experiment("credit_risk_modeling")
    
    data = load_data()
    X = data.drop('is_high_risk', axis=1)
    y = data['is_high_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Model candidates
    models = {
        "logistic": LogisticRegression(),
        "random_forest": RandomForestClassifier(),
        "gradient_boosting": GradientBoostingClassifier()
    }
    
    # Hyperparameter spaces
    param_spaces = {
        "logistic": {
            'C': hp.loguniform('C', -5, 5),
            'penalty': hp.choice('penalty', ['l2', 'none'])
        },
        "random_forest": {
            'n_estimators': hp.quniform('n_estimators', 50, 500, 50),
            'max_depth': hp.quniform('max_depth', 3, 10, 1)
        },
        "gradient_boosting": {
            'n_estimators': hp.quniform('n_estimators', 50, 500, 50),
            'learning_rate': hp.loguniform('learning_rate', -5, 0)
        }
    }
    
    best_models = {}
    
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            # Hyperparameter tuning
            def objective(params):
                clf = model.set_params(**params)
                clf.fit(X_train, y_train)
                preds = clf.predict_proba(X_test)[:, 1]
                score = roc_auc_score(y_test, preds)
                mlflow.log_metric("val_auc", score)
                return {'loss': -score, 'status': STATUS_OK}
            
            best = fmin(
                fn=objective,
                space=param_spaces[model_name],
                algo=tpe.suggest,
                max_evals=20,
                trials=Trials()
            )
            
            # Train best model
            best_model = model.set_params(**best)
            best_model.fit(X_train, y_train)
            
            # Log metrics
            test_preds = best_model.predict_proba(X_test)[:, 1]
            mlflow.log_metrics({
                "test_auc": roc_auc_score(y_test, test_preds),
                "test_f1": f1_score(y_test, best_model.predict(X_test))
            })
            
            # Log model
            mlflow.sklearn.log_model(best_model, model_name)
            best_models[model_name] = best_model
    
    return best_models

if __name__ == "__main__":
    train_model()