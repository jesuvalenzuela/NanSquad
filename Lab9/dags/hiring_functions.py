import os
from datetime import datetime
from pathlib import Path
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

def create_folders(**kwargs) -> dict:
    """
    Crea una carpeta principal que utiliza la fecha de ejecución del DAG como nombre.
    Dentro de esta carpeta se crean las subcarpetas:
        - raw
        - splits
        - models

    Parámetros:
    -----------
    **kwargs : dict
        Diccionario que puede contener variables templadas de Airflow,
        como 'ts_nodash' o 'ds_nodash'.

    Retorna:
    --------
    dict : Rutas creadas para cada carpeta.
    """

    # Obtiene la fecha de ejecución del DAG (si existe) o usa la fecha actual
    run_name = kwargs.get("ts_nodash", datetime.now().strftime("%Y%m%dT%H%M%S"))

    # Crea la ruta base runs/<fecha_ejecución>/
    base_dir = Path("runs") / run_name

    # Subcarpetas a crear
    subfolders = ["raw", "splits", "models"]

    # Crea las carpetas sin usar bucles explícitos
    list(map(lambda sf: (base_dir / sf).mkdir(parents=True, exist_ok=True), subfolders))

    # Retorna las rutas creadas
    return {
        "run_dir": str(base_dir),
        "raw_dir": str(base_dir / "raw"),
        "splits_dir": str(base_dir / "splits"),
        "models_dir": str(base_dir / "models"),
    }


TARGET_COL = "HiringDecision"

def split_data(
    raw_dir: str | None = None,
    splits_dir: str | None = None,
    test_size: float = 0.20,
    seed: int = 42,
    **kwargs
) -> dict:
    """
    Lee data_1.csv desde raw/, aplica hold-out estratificado y guarda
    train.csv y test.csv en splits/.

    Parámetros (opcionalmente templados desde Airflow):
    - raw_dir: ruta a la carpeta raw del run.
    - splits_dir: ruta a la carpeta splits del run.
    - test_size: fracción para test (default 0.20).
    - seed: semilla (default 42).
    - **kwargs: puede contener 'ts_nodash' (timestamp del run).

    Retorna:
    - dict con rutas a train.csv y test.csv
    """
    # Si no vienen rutas, las deducimos del timestamp del DAG o de ahora
    run_name = kwargs.get("ts_nodash", datetime.now().strftime("%Y%m%dT%H%M%S"))
    base = Path("runs") / str(run_name)
    raw_dir = Path(raw_dir) if raw_dir else base / "raw"
    splits_dir = Path(splits_dir) if splits_dir else base / "splits"

    # Paths
    csv_path = raw_dir / "data_1.csv"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Cargar datos
    df = pd.read_csv(csv_path)

    # Hold-out estratificado
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Guardar
    train_path = splits_dir / "train.csv"
    test_path = splits_dir / "test.csv"
    pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)

    return {"train_path": str(train_path), "test_path": str(test_path)}


# ==== 1.1.3: preprocesamiento + entrenamiento + guardado del pipeline ====
# columnas categóricas según el dataset del lab
CATEGORICAL = ["Gender", "EducationLevel", "RecruitmentStrategy"]
TARGET_COL = "HiringDecision"

def preprocess_and_train(
    train_csv: str | None = None,
    test_csv: str  | None = None,
    models_dir: str | None = None,
    **kwargs
) -> dict:
    """
    Lee train/test desde 'splits/', crea un Pipeline con ColumnTransformer y RandomForest,
    imprime Accuracy y F1 (clase positiva=1) en test, y guarda el pipeline con joblib en 'models/'.
    Si no se entregan rutas, se infieren usando ts_nodash (o ahora) bajo runs/<run>/.
    """
    # Inferir rutas si no se entregan explícitamente
    run_name = kwargs.get("ts_nodash", datetime.now().strftime("%Y%m%dT%H%M%S"))
    base = Path("runs") / str(run_name)
    splits = base / "splits"
    models = Path(models_dir) if models_dir else (base / "models")
    models.mkdir(parents=True, exist_ok=True)

    train_csv = Path(train_csv) if train_csv else (splits / "train.csv")
    test_csv  = Path(test_csv)  if test_csv  else (splits / "test.csv")

    # Cargar datos
    train = pd.read_csv(train_csv)
    test  = pd.read_csv(test_csv)

    X_train, y_train = train.drop(columns=[TARGET_COL]), train[TARGET_COL]
    X_test,  y_test  = test.drop(columns=[TARGET_COL]),  test[TARGET_COL]

    # Detectar numéricas (todo lo que no es categórico ni target)
    numerical = [c for c in X_train.columns if c not in CATEGORICAL]

    # Preprocesador: OneHot a categóricas, StandardScaler a numéricas
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", StandardScaler(), numerical),
        ],
        remainder="drop"
    )

    # Modelo
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    # Pipeline completo
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    # Entrenar
    pipe.fit(X_train, y_train)

    # Evaluar
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_pos = f1_score(y_test, y_pred, pos_label=1)

    print(f"Accuracy (test): {acc:.4f}")
    print(f"F1 (clase positiva=1): {f1_pos:.4f}")

    # Guardar pipeline entrenado
    model_path = models / "hiring_rf_pipeline.joblib"
    joblib.dump(pipe, model_path)

    return {"model_path": str(model_path), "accuracy": float(acc), "f1_positive": float(f1_pos)}

# ==== Interfaz solicitada: gradio_interface ====
def gradio_interface(model_path: str):
    """
    Interfaz mínima para probar el modelo entrenado desde 'models/'.
    Ejecuta por ejemplo:
      python -c "from dags.hiring_functions import launch_gradio; launch_gradio('runs/AAAA.../models/hiring_rf_pipeline.joblib')"
    """
    import gradio as gr

    pipe = joblib.load(model_path)

    def predict_fn(
        Age, Gender, EducationLevel, ExperienceYears, PreviousCompanies,
        DistanceFromCompany, InterviewScore, SkillScore, PersonalityScore, RecruitmentStrategy
    ):
        row = pd.DataFrame([{
            "Age": Age,
            "Gender": Gender,
            "EducationLevel": EducationLevel,
            "ExperienceYears": ExperienceYears,
            "PreviousCompanies": PreviousCompanies,
            "DistanceFromCompany": DistanceFromCompany,
            "InterviewScore": InterviewScore,
            "SkillScore": SkillScore,
            "PersonalityScore": PersonalityScore,
            "RecruitmentStrategy": RecruitmentStrategy
        }])
        proba = pipe.predict_proba(row)[:, 1].item()
        pred = int(proba >= 0.5)
        return {"Probabilidad Contratado": round(proba, 4), "Predicción (1=Sí)": pred}

    with gr.Blocks() as demo:
        gr.Markdown("## Hiring Decision – Random Forest")
        Age = gr.Slider(18, 60, value=30, step=1, label="Age")
        Gender = gr.Dropdown([0, 1], value=0, label="Gender")
        EducationLevel = gr.Dropdown([1, 2, 3, 4], value=2, label="EducationLevel")
        ExperienceYears = gr.Slider(0, 30, value=5, step=1, label="ExperienceYears")
        PreviousCompanies = gr.Slider(0, 10, value=2, step=1, label="PreviousCompanies")
        DistanceFromCompany = gr.Slider(0, 100, value=10.0, step=0.1, label="DistanceFromCompany")
        InterviewScore = gr.Slider(0, 100, value=50, step=1, label="InterviewScore")
        SkillScore = gr.Slider(0, 100, value=50, step=1, label="SkillScore")
        PersonalityScore = gr.Slider(0, 100, value=50, step=1, label="PersonalityScore")
        RecruitmentStrategy = gr.Dropdown([1, 2, 3], value=2, label="RecruitmentStrategy")

        out = gr.JSON(label="Resultado")
        gr.Button("Predecir").click(
            predict_fn,
            inputs=[Age, Gender, EducationLevel, ExperienceYears, PreviousCompanies,
                    DistanceFromCompany, InterviewScore, SkillScore, PersonalityScore, RecruitmentStrategy],
            outputs=out
        )
    return demo

def launch_gradio(model_path: str):
    demo = gradio_interface(model_path)
    demo.launch()