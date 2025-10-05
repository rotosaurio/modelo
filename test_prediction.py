#!/usr/bin/env python3
"""
Script de prueba rápido para verificar que la predicción funciona con datos sintéticos
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Configurar para usar solo datos sintéticos
os.environ['FORCE_SYNTHETIC'] = 'true'

from data_synthetic import SyntheticWeatherGenerator
from data_preprocessing import WeatherDataPreprocessor
from feature_engineering import WeatherFeatureEngineer
from weather_model import create_model, ModelTrainer
import joblib

def test_prediction():
    print("Probando predicción con datos sintéticos...")

    # 1. Crear datos sintéticos
    synthetic_gen = SyntheticWeatherGenerator()
    preprocessor = WeatherDataPreprocessor()
    feature_engineer = WeatherFeatureEngineer()

    # Coordenadas de Chihuahua
    lat, lon = 28.632, -106.069
    postal_code = "31125"

    # Generar datos sintéticos para las últimas 6 horas
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(hours=6)

    print(f"Generando datos sintéticos de {start_date} a {end_date}...")
    synthetic_data = synthetic_gen.generate_synthetic_weather(
        lat, lon, start_date.to_pydatetime(), end_date.to_pydatetime()
    )

    if synthetic_data.empty:
        print("ERROR: Error generando datos sintéticos")
        return False

    print(f"OK: Datos sintéticos generados: {len(synthetic_data)} registros")

    # 2. Crear características
    print("Creando características...")
    features = feature_engineer.create_all_features(synthetic_data)

    if features.empty:
        print("ERROR: Error creando características")
        return False

    print(f"OK: Características creadas: {len(features)} registros, {len(features.columns)} columnas")

    # 3. Seleccionar características para el modelo
    selected_features = feature_engineer.select_features_for_model(features)

    print(f"OK: Características seleccionadas: {len(selected_features)} registros, {len(selected_features.columns)} columnas")

    # 4. Cargar modelo entrenado
    print("Cargando modelo entrenado...")
    try:
        model = create_model(input_size=20, num_targets=6, model_type='simple', hidden_size=16, num_layers=1)
        trainer = ModelTrainer(model)

        # Intentar cargar el modelo guardado
        if not trainer.load_model():
            print("ERROR: Error cargando modelo entrenado")
            return False

        print("OK: Modelo cargado correctamente")

    except Exception as e:
        print(f"ERROR: Error con el modelo: {e}")
        return False

    # 5. Preparar datos para predicción
    print("Preparando datos para predicción...")
    try:
        # Crear dataset de predicción
        from dataset_creation import WeatherDatasetCreator
        dataset_creator = WeatherDatasetCreator()

        prediction_data = dataset_creator.create_prediction_dataset(lat, lon, postal_code)
        if prediction_data is None:
            print("ERROR: Error creando dataset de predicción")
            return False

        X_pred = prediction_data['X_pred']
        print(f"OK: Datos de predicción preparados: {X_pred.shape}")

    except Exception as e:
        print(f"ERROR: Error preparando datos: {e}")
        return False

    # 6. Hacer predicción
    print("Ejecutando predicción...")
    try:
        predictions = trainer.predict(X_pred)
        print(f"OK: Predicción completada: {predictions.shape}")
        print(f"Primeras predicciones: {predictions[0, :3, :]}")

        return True

    except Exception as e:
        print(f"ERROR: Error en predicción: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_prediction()
    if success:
        print("\nEXITO: Predicción exitosa! El modelo funciona correctamente.")
    else:
        print("\nERROR: Error en la predicción.")
        sys.exit(1)
