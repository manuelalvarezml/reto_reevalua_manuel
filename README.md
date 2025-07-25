# Clasificación de Riesgo Crediticio con SageMaker + Bedrock

Este repo mezcla machine learning tradicional con modelos generativos para predecir el riesgo crediticio. Creé descripciones generadas por AI (descriptions) y etiquetas inferidas (target) como features adicionales, y entrené un modelo final en SageMaker con XGBoost.

## 📁 Estructura del proyecto

```
├── artifacts/ # Transformadores guardados (OneHotEncoder, TF-IDF)
├── auxiliar_scripts/ # Scripts auxiliares para regenerar errores
├── bedrock_test_files/ # Pruebas para ver modelos disponibles de Amazon Bedrock y probar conectividad
├── data_files/ # Dataset inicial, con 'descriptions' y dataset agregado con 'target'
├── eda_credit_risk.ipynb # Notebook de exploración de datos
├── generate_descriptions.py # Genera descripciones 'descriptions' con Bedrock (genAI)
├── generate_risk_targets.py # Genera etiquetas 'target' (good/bad risk) con Bedrock
├── preprocess_for_sagemaker.py # Preprocesa y exporta CSVs (train, test sets) para usar SageMaker
├── train_model_sagemaker.py # Lanza el job de entrenamiento en AWS SageMaker
├── train_data_sagemaker.csv # Set de entrenamiento
├── test_data_sagemaker.csv # Set de validación
├── y_test_true_labels.csv # Etiquetas reales para evaluar el modelo
```

## ⚙️ Flujo del pipeline

1. **Generar descripciones y etiquetas con AI**  
   Corremos `generate_descriptions.py` y `generate_risk_targets.py` usando Bedrock para enriquecer los datos.

2. **Preprocesar para SageMaker**  
   `preprocess_for_sagemaker.py` convierte todo a features numéricos (one-hot + TF-IDF) y guarda los CSVs.

3. **Entrenar en SageMaker**  
   `train_model_sagemaker.py` sube los datos a S3 y lanza un entrenamiento con XGBoost directamente en la nube.

4. **Evaluar resultados**  
   Las etiquetas verdaderas están en `y_test_true_labels.csv` para comparar después contra las predicciones.

## ✅ Requisitos

- Python 3.8+
- AWS CLI + permisos para usar SageMaker y Bedrock
- Tener acceso habilitado a Amazon Bedrock
