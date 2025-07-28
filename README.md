# 🏦 Clasificación de Riesgo Crediticio con SageMaker + Bedrock

Este proyecto combina modelos generativos (Amazon Bedrock) con machine learning tradicional para predecir riesgo crediticio. Se generan descripciones (`description`) y etiquetas (`target`) con GenAI y se entrena un modelo en SageMaker (Logistic Regression) para inferir riesgo de crédito.

---

## 📁 Estructura del Proyecto

```
├── artifacts/                      # Preprocesadores guardados (OneHotEncoder)
├── downloaded_artifacts/          # Artifacts descargados del modelo entrenado
├── data_files/                    # Dataset original + columnas generadas (description, target)
├── bedrock_test_files/            # Pruebas para conexión y modelos disponibles de Bedrock
├── train_data_sagemaker.csv       # Datos de entrenamiento listos para SageMaker
├── test_data_sagemaker.csv        # Datos de test para evaluación
├── y_test_true_labels.csv         # Etiquetas verdaderas para el test
├── test_data_for_inference.csv    # Archivo de prueba para hacer inferencia en el endpoint
├── eda_credit_risk.ipynb          # Exploración de datos (EDA)
├── generate_descriptions.py       # Usa Bedrock para crear la columna 'description'
├── generate_risk_targets.py       # Usa Bedrock para generar la columna 'target' (good/bad risk)
├── preprocess_for_sagemaker.py    # Preprocesa y guarda archivos para SageMaker (OneHot)
├── train_logreg_sagemaker.py      # Entrena regresión logística con sklearn en la nube
├── deploy_model_sagemaker.py      # Despliega el endpoint en SageMaker
├── invoke_endpoint.py             # Realiza inferencia en el endpoint
```

---

## ⚙️ Flujo del pipeline (Logistic Regression)

1. **Preprocesamiento:**

   ```bash
   python preprocess_for_sagemaker.py
   ```

   Genera:

   - `train_data_sagemaker.csv`, `test_data_sagemaker.csv`, `y_test_true_labels.csv`
   - `artifacts/tabular_preprocessor.joblib`

2. **Subida a S3:**

   ```bash
   python upload_dataset_to_s3.py
   ```

3. **Entrenamiento:**

   ```bash
   python train_logreg_sagemaker.py
   ```

4. **Descarga artifacts del job:**

   - Actualiza `job_name` en `download_artifacts_logreg_training.py`

   ```bash
   python download_artifacts_logreg_training.py
   ```

5. **Empaquetado para despliegue:**

   ```bash
   cp artifacts/tabular_preprocessor.joblib downloaded_artifacts/
   cd downloaded_artifacts
   tar -czf model.tar.gz model.joblib train_logreg.py tabular_preprocessor.joblib
   ```

6. **Despliegue del modelo:**

   ```bash
   python upload_model_to_s3_for_deployment.py
   python deploy_model_sagemaker.py
   ```

7. **Inferencia:**
   ```bash
   python invoke_endpoint.py
   ```

---

## 📊 Resultados

Se entrenó un modelo de regresión logística con `class_weight="balanced"` para mitigar el desbalance de clases. Esto mejoró considerablemente el recall en los casos de **alto riesgo** (de 32.6% a 76.1%) a cambio de una leve caída en accuracy.

**Métricas:**

- Accuracy: 70%
- Recall (bad risk): **76.1%**
- Precision (good risk): 90.5%
- ROC AUC: **0.7974**

📌 Se prioriza **minimizar falsos negativos en riesgos malos** (i.e. no dar préstamos a quien no debe).

---

## 🧪 Inferencia de prueba

Se probó con 15 casos:

- 5 de **riesgo alto obvio** → correctamente clasificados como 0
- 5 de **riesgo bajo obvio** → correctamente clasificados como 1
- 5 mixtos → desempeño razonable

✅ El modelo predice correctamente según el objetivo del negocio: protegerse ante perfiles de alto riesgo.

---

## ✅ Requisitos

- Python 3.8
- `scikit-learn` entre **1.0 y 1.2** (límite actual en SageMaker: [ver documentación oficial](https://docs.aws.amazon.com/sagemaker/latest/dg/sklearn.html))
- AWS CLI + permisos para usar:
  - Amazon SageMaker
  - Amazon Bedrock (activado)
