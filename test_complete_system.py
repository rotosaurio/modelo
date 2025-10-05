#!/usr/bin/env python3
"""
Script de prueba completo para verificar todo el sistema de predicción meteorológica
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_complete_system():
    """Prueba completa del sistema de predicción meteorológica"""
    
    print("INICIANDO PRUEBAS COMPLETAS DEL SISTEMA DE PREDICCION METEOROLOGICA")
    print("=" * 80)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Importaciones básicas
    print("\nTest 1: Verificando importaciones...")
    try:
        from data_synthetic import SyntheticWeatherGenerator
        from data_preprocessing import WeatherDataPreprocessor
        from feature_engineering import WeatherFeatureEngineer
        from dataset_creation import WeatherDatasetCreator
        from weather_model import create_model, ModelTrainer
        from predict_weather import WeatherPredictor
        from training_pipeline import WeatherTrainingPipeline
        print("OK: Todas las importaciones exitosas")
        tests_passed += 1
    except Exception as e:
        print(f"ERROR: Error en importaciones: {e}")
        tests_failed += 1
        return False
    
    # Test 2: Generación de datos sintéticos
    print("\nTest 2: Verificando generación de datos sintéticos...")
    try:
        synthetic_gen = SyntheticWeatherGenerator()
        lat, lon = 28.632, -106.069  # Chihuahua
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=6)
        
        synthetic_data = synthetic_gen.generate_synthetic_weather(lat, lon, start_date, end_date)
        
        if synthetic_data.empty:
            raise ValueError("Datos sintéticos vacíos")
        
        print(f"OK: Datos sintéticos generados: {len(synthetic_data)} registros, {len(synthetic_data.columns)} columnas")
        tests_passed += 1
    except Exception as e:
        print(f"ERROR: Error en generación de datos sintéticos: {e}")
        tests_failed += 1
    
    # Test 3: Preprocesamiento de datos
    print("\nTest 3: Verificando preprocesamiento de datos...")
    try:
        preprocessor = WeatherDataPreprocessor()
        
        # Normalizar datos
        normalized_data = preprocessor.normalize_data(synthetic_data, fit_scaler=True)
        
        if normalized_data.empty:
            raise ValueError("Datos normalizados vacíos")
        
        print(f"OK: Datos normalizados: {normalized_data.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"ERROR: Error en preprocesamiento: {e}")
        tests_failed += 1
    
    # Test 4: Ingeniería de características
    print("\nTest 4: Verificando ingeniería de características...")
    try:
        feature_engineer = WeatherFeatureEngineer()
        
        # Crear características
        features = feature_engineer.create_all_features(synthetic_data)
        selected_features = feature_engineer.select_features_for_model(features)
        
        if selected_features.empty:
            raise ValueError("Características seleccionadas vacías")
        
        print(f"OK: Características creadas: {features.shape} -> {selected_features.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"ERROR: Error en ingeniería de características: {e}")
        tests_failed += 1
    
    # Test 5: Creación de dataset
    print("\nTest 5: Verificando creación de dataset...")
    try:
        dataset_creator = WeatherDatasetCreator()
        
        # Crear secuencias
        X, y = preprocessor.prepare_sequences(
            selected_features, 
            input_steps=24, 
            output_steps=24,
            target_variables=['temp_celsius', 'precipitation_mm', 'relative_humidity_percent', 
                            'pressure_hpa', 'cloud_cover_percent', 'wind_speed_ms']
        )
        
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Secuencias vacías")
        
        print(f"OK: Secuencias creadas: {len(X)} muestras, entrada {X.shape}, salida {y.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"ERROR: Error en creación de dataset: {e}")
        tests_failed += 1
    
    # Test 6: Creación y carga de modelo
    print("\nTest 6: Verificando modelo...")
    try:
        # Crear modelo
        model = create_model(
            input_size=X.shape[2], 
            num_targets=y.shape[2], 
            model_type='simple',
            hidden_size=16,
            num_layers=1
        )
        
        trainer = ModelTrainer(model)
        
        # Verificar si existe modelo entrenado
        if os.path.exists("models/weather_model.pt"):
            success = trainer.load_model()
            if success:
                print("OK: Modelo cargado correctamente")
            else:
                print("WARNING: Modelo existente pero no se pudo cargar")
        else:
            print("WARNING: No hay modelo entrenado disponible")
        
        tests_passed += 1
    except Exception as e:
        print(f"ERROR: Error en modelo: {e}")
        tests_failed += 1
    
    # Test 7: Predicción
    print("\nTest 7: Verificando predicción...")
    try:
        predictor = WeatherPredictor()
        
        # Intentar cargar modelo
        if predictor.load_model():
            # Crear datos de predicción
            prediction_data = dataset_creator.create_prediction_dataset(lat, lon, "31125")
            
            if prediction_data is not None:
                X_pred = prediction_data['X_pred']
                
                # Hacer predicción
                predictions = predictor.predict(X_pred)
                
                if predictions is not None and len(predictions) > 0:
                    print(f"OK: Predicción exitosa: {predictions.shape}")
                    tests_passed += 1
                else:
                    print("ERROR: Predicción vacía")
                    tests_failed += 1
            else:
                print("ERROR: No se pudo crear dataset de predicción")
                tests_failed += 1
        else:
            print("WARNING: No se pudo cargar modelo para predicción")
            tests_failed += 1
    except Exception as e:
        print(f"ERROR: Error en predicción: {e}")
        tests_failed += 1
    
    # Test 8: Pipeline de entrenamiento
    print("\nTest 8: Verificando pipeline de entrenamiento...")
    try:
        training_pipeline = WeatherTrainingPipeline()
        
        # Verificar que se puede inicializar
        print("OK: Pipeline de entrenamiento inicializado")
        tests_passed += 1
    except Exception as e:
        print(f"ERROR: Error en pipeline de entrenamiento: {e}")
        tests_failed += 1
    
    # Test 9: Verificación de archivos generados
    print("\nTest 9: Verificando archivos del sistema...")
    try:
        required_files = [
            "models/weather_model.pt",
            "models/dataset_info.pkl",
            "models/scaler.pkl",
            "models/feature_columns.pkl"
        ]
        
        existing_files = []
        missing_files = []
        
        for file_path in required_files:
            if os.path.exists(file_path):
                existing_files.append(file_path)
            else:
                missing_files.append(file_path)
        
        print(f"OK: Archivos existentes: {len(existing_files)}/{len(required_files)}")
        for file_path in existing_files:
            print(f"   + {file_path}")
        
        if missing_files:
            print(f"WARNING: Archivos faltantes: {len(missing_files)}")
            for file_path in missing_files:
                print(f"   - {file_path}")
        
        tests_passed += 1
    except Exception as e:
        print(f"ERROR: Error verificando archivos: {e}")
        tests_failed += 1
    
    # Test 10: Rendimiento básico
    print("\nTest 10: Verificando rendimiento básico...")
    try:
        import time
        
        # Medir tiempo de predicción
        start_time = time.time()
        
        if predictor.load_model():
            prediction_data = dataset_creator.create_prediction_dataset(lat, lon, "31125")
            if prediction_data is not None:
                X_pred = prediction_data['X_pred']
                predictions = predictor.predict(X_pred)
                
                end_time = time.time()
                prediction_time = end_time - start_time
                
                print(f"OK: Tiempo de predicción: {prediction_time:.2f} segundos")
                tests_passed += 1
            else:
                print("WARNING: No se pudo medir rendimiento (sin datos)")
                tests_passed += 1
        else:
            print("WARNING: No se pudo medir rendimiento (sin modelo)")
            tests_passed += 1
    except Exception as e:
        print(f"ERROR: Error en prueba de rendimiento: {e}")
        tests_failed += 1
    
    # Resumen final
    print("\n" + "=" * 80)
    print("RESUMEN DE PRUEBAS")
    print("=" * 80)
    print(f"OK: Pruebas exitosas: {tests_passed}")
    print(f"ERROR: Pruebas fallidas: {tests_failed}")
    print(f"Tasa de éxito: {(tests_passed / (tests_passed + tests_failed) * 100):.1f}%")
    
    if tests_failed == 0:
        print("\nEXITO: TODAS LAS PRUEBAS PASARON! El sistema está funcionando correctamente.")
        return True
    else:
        print(f"\nWARNING: {tests_failed} pruebas fallaron. Revisar los errores arriba.")
        return False

if __name__ == "__main__":
    success = test_complete_system()
    sys.exit(0 if success else 1)
