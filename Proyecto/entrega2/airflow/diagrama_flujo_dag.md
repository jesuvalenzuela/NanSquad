# Diagrama de Flujo - Pipeline de Predicción de Productos Prioritarios

## Diagrama Principal

```mermaid
flowchart TD
    Start([Start Pipeline]) --> CheckHist{Check Historical<br/>Data}
    
    %% Primera bifurcación: ¿Existe histórico?
    CheckHist -->|No existe| CopyRaw[Copy Raw<br/>to Historical]
    CheckHist -->|Existe| Pass1[Pass 1<br/>Empty]
    
    %% Ambas ramas convergen
    CopyRaw --> CheckNew{Check New<br/>Data}
    Pass1 --> CheckNew
    
    %% Segunda bifurcación: ¿Hay datos nuevos?
    CheckNew -->|Hay nuevos| ExtendDS[Extend Dataset<br/>Agregar nuevas filas]
    CheckNew -->|No hay nuevos| Pass2[Pass 2<br/>Empty]
    
    %% Ambas ramas convergen
    ExtendDS --> DecideTrain[Decide Training<br/>¿Se ejecutó copy_raw<br/>o extend_dataset?]
    Pass2 --> DecideTrain
    
    %% Tercera bifurcación: ¿Entrenar modelo?
    DecideTrain -->|Sí entrenar| PrepData[Prepare Data<br/>Limpieza y agregación]
    DecideTrain -->|No entrenar| NotTrain[Not Train<br/>Empty]
    
    %% Flujo de entrenamiento
    PrepData --> Split[Split Data<br/>Train/Val/Test]
    Split --> Preproc[Preprocess Data<br/>Pipeline de preprocesamiento]
    Preproc --> Optimize[Optimize Model<br/>Optuna + MLflow]
    
    %% Bifurcación en paralelo
    Optimize --> Evaluate[Evaluate and Interpret<br/>SHAP + MLflow]
    Optimize --> TrainFinal[Train Final Model<br/>Con todos los datos]

    %% Convergencia final
    Evaluate --> End([End Pipeline])
    TrainFinal --> End
    NotTrain --> End
    
    %% Estilos
    classDef startEnd fill:#90EE90,stroke:#333,stroke-width:3px
    classDef decision fill:#FFD700,stroke:#333,stroke-width:2px
    classDef process fill:#87CEEB,stroke:#333,stroke-width:2px
    classDef empty fill:#D3D3D3,stroke:#333,stroke-width:1px
    classDef important fill:#FF6B6B,stroke:#333,stroke-width:2px

    class Start,End startEnd
    class CheckHist,CheckNew,DecideTrain decision
    class PrepData,Split,Preproc,Optimize,Evaluate,TrainFinal process
    class Pass1,Pass2,NotTrain empty
    class CopyRaw,ExtendDS important
```

## Leyenda de Colores

- 🟢 **Verde**: Inicio y fin del pipeline
- 🟡 **Amarillo**: Puntos de decisión (branching)
- 🔵 **Azul**: Tareas de procesamiento y modelado
- ⚪ **Gris**: Operadores vacíos (pass)
- 🔴 **Rojo**: Tareas críticas de datos (copy_raw, extend_dataset)

## Flujos Posibles

### Escenario 1: Primera Ejecución
```
Start → Check Historical (no existe) → Copy Raw → Check New (no hay) →
Pass 2 → Decide Training (sí) → Prepare Data → Split → Preprocess →
Optimize → [Evaluate + Train Final] → End
```

### Escenario 2: Ejecución Regular con Datos Nuevos
```
Start → Check Historical (existe) → Pass 1 → Check New (hay nuevos) →
Extend Dataset → Decide Training (sí) → Prepare Data → Split → Preprocess →
Optimize → [Evaluate + Train Final] → End
```

### Escenario 3: Ejecución Regular sin Datos Nuevos
```
Start → Check Historical (existe) → Pass 1 → Check New (no hay) →
Pass 2 → Decide Training (no) → Not Train → End
```

## Puntos Clave del Diseño

1. **Tres decisiones principales**:
   - ¿Existe dataset histórico? (primera vez vs. ejecuciones posteriores)
   - ¿Hay datos nuevos? (reentrenamiento necesario)
   - ¿Entrenar modelo? (basado en las dos decisiones anteriores)

2. **Paralelización**:
   - `evaluate_and_interpret` y `train_final_model` se ejecutan en paralelo después de `optimize_model`

3. **Predicciones on-demand**:
   - Las predicciones se generan a través de la aplicación web, no en el DAG
   - El DAG se enfoca exclusivamente en entrenamiento y reentrenamiento del modelo

4. **Trigger Rules**:
   - `decide_training` usa `none_failed` para ejecutarse si cualquier rama upstream tuvo éxito
   - `end_pipeline` también usa `none_failed` para ejecutarse siempre
```

---

## Diagrama Simplificado (Alto Nivel)

```mermaid
flowchart LR
    A[📥 Inicio] --> B[🔍 Gestión<br/>de Datos]
    B --> C{¿Entrenar?}
    C -->|Sí| D[⚙️ Preparación<br/>de Datos]
    C -->|No| H[✅ Fin]
    D --> F[🎯 Optimización<br/>+ Evaluación]
    F --> H

    classDef phase fill:#4A90E2,stroke:#333,color:#fff
    class B,D,F phase
```
