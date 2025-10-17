import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
import mlflow
import os
from datetime import datetime
import pickle


def get_best_model(experiment_id):
    """CORRECCIÓN: Se agrega ascending=False para obtener el MEJOR modelo (mayor F1)"""
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.valid_f1", ascending=False)["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")

    return best_model   

def optimize_model(X_train, X_test, y_train, y_test, n_trials=50):
    """
    Optimiza XGBoost con Optuna y registra en MLFlow.
    """
    import logging
    logging.getLogger('mlflow').setLevel(logging.ERROR)

    # Crear directorios para guardar resultados
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Guardar versiones de librerías
    import sys
    with open("library_versions.txt", "w") as f:
        f.write(f"python: {sys.version}\n")
        f.write(f"optuna: {optuna.__version__}\n")
        f.write(f"mlflow: {mlflow.__version__}\n")
        f.write(f"xgboost: {xgb.__version__}\n")
        f.write(f"pandas: {pd.__version__}\n")
        f.write(f"matplotlib: {plt.matplotlib.__version__}\n")
        f.write(f"sklearn: {Pipeline.__module__.split('.')[0]}\n")
    
    # Configurar experimento MLFlow: en lugar de usar nombre default, agregamos fecha_hora para que sea reconocible
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"XGBoost_Optimization_{timestamp}"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id        # para get_best_model()

    # Función objetivo para Optuna
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'max_leaves': trial.suggest_int('max_leaves', 0, 100),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
            'random_state': 42
        }
        
        # Nombre interpretable para el run
        run_name = f"XGBoost_lr{params['learning_rate']:.3f}_depth{params['max_depth']}_n{params['n_estimators']}"
        
        # Registrar en MLFlow
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("mlflow.runName", run_name)

            # Pipeline con pre-procesamiento y modelo
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean').set_output(transform="pandas")),        # Imputamos dado que se identificó Nan
                ('cls', xgb.XGBClassifier(**params))
            ])
            
            # Entrenar y predecir
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='weighted')

            mlflow.log_params(params)
            mlflow.log_metric("valid_f1", f1)
            mlflow.sklearn.log_model(pipeline, 
                                     name="model")
        
        return f1
    
    # Ejecutar optimización
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Obtener mejor modelo usando get_best_model()
    best_model = get_best_model(experiment_id)
    
    # Serializar con pickle.dump y guardar en /models
    model_path = "models/best_xgboost_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    # Guardar gráficos de Optuna en /plots
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.write_image("plots/optimization_history.png")
    
    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.write_image("plots/param_importances.png")
    
    # Respaldar configuración del modelo final
    config_file = "plots/model_configuration.txt"
    with open(config_file, "w") as f:
        f.write("CONFIGURACIÓN DEL MODELO FINAL\n")
        f.write(f"F1-Score: {study.best_value:.4f}\n\n")
        f.write("Hiperparámetros:\n")
        for param, value in study.best_params.items():
            f.write(f"{param}: {value}\n")
    
    # Guardar importancia de variables en gráfico
    feature_importance = best_model.named_steps["cls"].feature_importances_
    feature_names = best_model.named_steps["cls"].get_booster().feature_names

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    y_pos = range(len(importance_df))
    plt.figure(figsize=(10, 6))
    plt.barh(y_pos, importance_df['Importance'])
    plt.yticks(y_pos, importance_df['Feature'])
    plt.xlabel('Importancia')
    plt.ylabel('Variable')
    plt.title('Importancia de Variables')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', bbox_inches='tight')
    plt.close()
    
    # Registrar artefactos en MLFlow
    with mlflow.start_run(run_name="Best_Model_Final"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("valid_f1", study.best_value)
        mlflow.log_metric("n_trials", n_trials)
        mlflow.log_artifacts("plots", artifact_path="plots")
        mlflow.log_artifact(model_path, artifact_path="models")
        mlflow.sklearn.log_model(best_model, name="final_model")
    
    return best_model

if __name__ == "__main__":

    # Cargar datos
    df = pd.read_csv('water_potability.csv')

    # Definir features y target
    X = df.drop(columns=['Potability'])
    y = df['Potability']

    # Separar conjunto de train y test 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
        
    # Ejecutar optimización
    best_model = optimize_model(
        X_train, X_test, y_train, y_test,
        n_trials=30
    )
    