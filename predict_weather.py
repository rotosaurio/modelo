"""
Script principal para predicción meteorológica en códigos postales de Chihuahua
Contiene la función get_weather_forecast() para obtener pronósticos
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import joblib

from config import (
    MODEL_SAVE_PATH, SCALER_SAVE_PATH, FEATURE_COLUMNS_SAVE_PATH,
    WEATHER_VARIABLES, WEATHER_THRESHOLDS, TIME_STEP_MINUTES, OUTPUT_STEPS,
    MODEL_CONFIG
)
from postal_coordinates import PostalCoordinatesManager
from dataset_creation import WeatherDatasetCreator
from weather_model import create_model, ModelTrainer
from data_preprocessing import WeatherDataPreprocessor
from feature_engineering import WeatherFeatureEngineer

logger = logging.getLogger(__name__)


class WeatherPredictor:
    """Clase principal para hacer predicciones meteorológicas"""

    def __init__(self):
        self.coord_manager = PostalCoordinatesManager()
        self.dataset_creator = WeatherDatasetCreator()
        self.preprocessor = WeatherDataPreprocessor()
        self.feature_engineer = WeatherFeatureEngineer()
        self.model = None
        self.trainer = None
        self.is_loaded = False

        # Configurar logging básico
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    def load_model(self) -> bool:
        """
        Carga el modelo entrenado y todos los artefactos necesarios
        Returns:
            True si se cargó correctamente
        """
        try:
            logger.info("Cargando modelo y artefactos...")

            # Verificar que existan los archivos necesarios
            if not MODEL_SAVE_PATH.exists():
                logger.error(f"Archivo de modelo no encontrado: {MODEL_SAVE_PATH}")
                logger.error("Solución: Entrena el modelo primero con 'python main.py train'")
                return False

            if not SCALER_SAVE_PATH.exists():
                logger.error(f"Archivo de escalador no encontrado: {SCALER_SAVE_PATH}")
                logger.error("Solución: Reentrena el modelo con 'python main.py train --force-retrain'")
                return False

            # Cargar información del dataset
            dataset_info = self.dataset_creator.load_dataset_info()
            if dataset_info is None:
                logger.error("No se pudo cargar información del dataset")
                logger.error("Solución: Reentrena el modelo con 'python main.py train --force-retrain'")
                return False

            # Cargar checkpoint para obtener hiperparámetros
            import torch
            checkpoint = torch.load(MODEL_SAVE_PATH, map_location='cpu')
            
            # Obtener hiperparámetros del modelo guardado
            if 'hyperparams' in checkpoint:
                hyperparams = checkpoint['hyperparams']
                n_features = hyperparams['input_size']
                n_targets = hyperparams['num_targets']
                model_type = 'simple' if hyperparams['model_class'] == 'WeatherPredictorSimplified' else 'full'
                logger.info(f"Hiperparámetros cargados: input_size={n_features}, hidden_size={hyperparams['hidden_size']}, model_type={model_type}")
            else:
                # Fallback a dataset_info
                n_features = dataset_info.get('n_features', 20)
                n_targets = dataset_info.get('n_targets', 6)
                model_type = 'simple'
                hidden_size = MODEL_CONFIG['hidden_size'] // 2  # 16 para modelo simple
                num_layers = 1
                logger.warning("No se encontraron hiperparámetros en el checkpoint, usando valores por defecto")

            # Crear modelo con la arquitectura correcta
            hidden_size = hyperparams.get('hidden_size', MODEL_CONFIG['hidden_size'] // 2 if model_type == 'simple' else MODEL_CONFIG['hidden_size'])
            num_layers = hyperparams.get('num_layers', MODEL_CONFIG['num_layers'])

            self.model = create_model(
                input_size=n_features,
                num_targets=n_targets,
                model_type=model_type,
                hidden_size=hidden_size,
                num_layers=num_layers
            )

            # Crear trainer y cargar modelo
            self.trainer = ModelTrainer(self.model)
            if not self.trainer.load_model():
                logger.error("Error cargando el modelo entrenado")
                logger.error("Solución: Reentrena el modelo con 'python main.py train --force-retrain'")
                return False

            # Cargar escalador
            if SCALER_SAVE_PATH.exists():
                self.preprocessor.scaler = joblib.load(SCALER_SAVE_PATH)
                logger.info(f"Escalador cargado: {SCALER_SAVE_PATH}")
            else:
                logger.warning(f"Archivo de escalador no encontrado: {SCALER_SAVE_PATH}")
                self.preprocessor.scaler = None

            # Cargar columnas de características
            if FEATURE_COLUMNS_SAVE_PATH.exists():
                try:
                    self.preprocessor.feature_columns = joblib.load(FEATURE_COLUMNS_SAVE_PATH)
                except Exception as e:
                    logger.warning(f"Error cargando columnas de características de {FEATURE_COLUMNS_SAVE_PATH}: {e}. Usando lista vacía.")
                    self.preprocessor.feature_columns = []
                logger.info(f"Columnas de features cargadas: {len(self.preprocessor.feature_columns)} columnas")
            else:
                logger.warning(f"Archivo de columnas no encontrado: {FEATURE_COLUMNS_SAVE_PATH}")
                self.preprocessor.feature_columns = []

            logger.info("Modelo y artefactos cargados correctamente")
            self.is_loaded = True
            return True

        except RuntimeError as e:
            if "size mismatch" in str(e):
                logger.error(f"Error de dimensiones: {e}")
                logger.error("El modelo guardado no coincide con la configuración actual")
                logger.error("Solución: Reentrena el modelo con 'python main.py train --force-retrain'")
            else:
                logger.error(f"Error cargando modelo: {e}")
            return False
        except Exception as e:
            logger.error(f"Error inesperado cargando modelo: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def get_weather_forecast(self, postal_code: str) -> Dict[str, Any]:
        """
        Función principal para obtener pronóstico meteorológico
        Args:
            postal_code: Código postal de Chihuahua (5 dígitos)
        Returns:
            Diccionario con pronóstico completo
        """
        try:
            # Validar entrada
            if not isinstance(postal_code, str) or len(postal_code) != 5:
                return {
                    "error": "Código postal inválido. Debe ser un string de 5 dígitos.",
                    "postal_code": postal_code
                }

            # Cargar modelo si no está cargado
            if not self.is_loaded and not self.load_model():
                return {
                    "error": "No se pudo cargar el modelo de predicción.",
                    "postal_code": postal_code
                }

            logger.info(f"Generando pronóstico para código postal: {postal_code}")

            # 1. Obtener coordenadas
            coords = self.coord_manager.get_coordinates(postal_code)
            if coords is None:
                return {
                    "error": f"No se encontraron coordenadas para el código postal {postal_code}",
                    "postal_code": postal_code
                }

            lat, lon = coords
            logger.info(f"Coordenadas encontradas: ({lat:.3f}, {lon:.3f})")

            # 2. Preparar datos para predicción
            prediction_data = self.dataset_creator.create_prediction_dataset(lat, lon, postal_code)
            if prediction_data is None:
                return {
                    "error": "No se pudieron obtener datos recientes para predicción",
                    "postal_code": postal_code,
                    "coords": [lat, lon]
                }

            # 3. Hacer predicción
            X_pred = prediction_data['X_pred']
            predictions_normalized = self.trainer.predict(X_pred)

            # 4. Desnormalizar predicciones
            predictions_denorm = self._denormalize_predictions(predictions_normalized, prediction_data)

            # 5. Formatear resultado
            result = self._format_forecast_result(
                postal_code, lat, lon, predictions_denorm,
                prediction_data['timestamp']
            )

            logger.info(f"Pronóstico generado exitosamente para {postal_code}")
            return result

        except Exception as e:
            logger.error(f"Error generando pronóstico: {e}")
            return {
                "error": f"Error interno del sistema: {str(e)}",
                "postal_code": postal_code
            }

    def _denormalize_predictions(self, predictions_normalized: np.ndarray,
                               prediction_data: Dict[str, Any]) -> np.ndarray:
        """
        Desnormaliza las predicciones del modelo
        Args:
            predictions_normalized: Predicciones normalizadas (batch_size, output_steps, n_targets)
            prediction_data: Datos de predicción
        Returns:
            Predicciones desnormalizadas
        """
        try:
            batch_size, output_steps, n_targets = predictions_normalized.shape
            
            # Verificar si tenemos escalador disponible
            if self.preprocessor.scaler is None:
                logger.warning("No hay escalador disponible, usando valores por defecto")
                # Aplicar transformación inversa básica basada en rangos típicos
                return self._apply_default_denormalization(predictions_normalized)

            # Crear DataFrame con las mismas columnas que se usaron en el entrenamiento
            target_columns = self.dataset_creator.target_variables
            
            # Si no tenemos las columnas exactas, usar las del escalador
            if not target_columns or len(target_columns) != n_targets:
                target_columns = self.preprocessor.feature_columns[:n_targets] if self.preprocessor.feature_columns else [f"target_{i}" for i in range(n_targets)]
            
            pred_df = pd.DataFrame(
                predictions_normalized.reshape(-1, n_targets),
                columns=target_columns
            )

            # Desnormalizar
            pred_denorm_df = self.preprocessor.denormalize_data(pred_df)

            # Reconstruir array 3D
            predictions_denorm = pred_denorm_df.values.reshape(batch_size, output_steps, n_targets)

            return predictions_denorm

        except Exception as e:
            logger.error(f"Error desnormalizando predicciones: {e}")
            # Fallback a desnormalización básica
            return self._apply_default_denormalization(predictions_normalized)

    def _apply_default_denormalization(self, predictions_normalized: np.ndarray) -> np.ndarray:
        """
        Aplica desnormalización básica usando rangos típicos de variables meteorológicas
        """
        try:
            # Rangos típicos para variables meteorológicas (después de normalización estándar)
            # Estos valores son aproximaciones basadas en datos meteorológicos típicos
            
            # Variables en orden: temp, precipitation, humidity, wind_speed, pressure, cloud_cover
            typical_ranges = {
                'temp': (15, 5),      # Media: 15°C, Std: 5°C
                'precipitation': (0.5, 2.0),  # Media: 0.5 mm, Std: 2.0 mm
                'humidity': (60, 20),  # Media: 60%, Std: 20%
                'wind_speed': (3, 2),  # Media: 3 m/s, Std: 2 m/s
                'pressure': (1013, 10), # Media: 1013 hPa, Std: 10 hPa
                'cloud_cover': (50, 30)  # Media: 50%, Std: 30%
            }
            
            predictions_denorm = predictions_normalized.copy()
            batch_size, output_steps, n_targets = predictions_denorm.shape
            
            # Aplicar transformación inversa de normalización estándar: x_original = x_norm * std + mean
            for i, (var_name, (mean, std)) in enumerate(typical_ranges.items()):
                if i < n_targets:
                    # Desnormalizar
                    predictions_denorm[:, :, i] = predictions_denorm[:, :, i] * std + mean
                    
                    # Aplicar límites físicamente realistas
                    if var_name == 'precipitation':
                        predictions_denorm[:, :, i] = np.maximum(0, predictions_denorm[:, :, i])  # No negativa
                    elif var_name == 'humidity':
                        predictions_denorm[:, :, i] = np.clip(predictions_denorm[:, :, i], 0, 100)  # 0-100%
                    elif var_name == 'wind_speed':
                        predictions_denorm[:, :, i] = np.maximum(0, predictions_denorm[:, :, i])  # No negativa
                    elif var_name == 'cloud_cover':
                        predictions_denorm[:, :, i] = np.clip(predictions_denorm[:, :, i], 0, 100)  # 0-100%
                    elif var_name == 'pressure':
                        predictions_denorm[:, :, i] = np.maximum(800, predictions_denorm[:, :, i])  # Mínimo 800 hPa
                    # temp puede ser negativa (bajo cero)
            
            logger.info("Desnormalización básica aplicada con rangos típicos")
            return predictions_denorm
            
        except Exception as e:
            logger.error(f"Error en desnormalización básica: {e}")
            return predictions_normalized

    def _format_forecast_result(self, postal_code: str, lat: float, lon: float,
                              predictions: np.ndarray, base_timestamp: datetime) -> Dict[str, Any]:
        """
        Formatea el resultado del pronóstico en el formato solicitado
        Args:
            postal_code: Código postal
            lat: Latitud
            lon: Longitud
            predictions: Array de predicciones (1, output_steps, n_targets)
            base_timestamp: Timestamp base para las predicciones
        Returns:
            Diccionario formateado con el pronóstico
        """
        try:
            # Variables objetivo en orden
            target_vars = self.dataset_creator.target_variables

            # Crear lista de pronósticos por paso temporal
            forecast = []
            predictions = predictions[0]  # Remover dimensión batch (siempre 1)

            for step in range(OUTPUT_STEPS):
                # Calcular timestamp para este paso
                step_timestamp = base_timestamp + timedelta(minutes=(step + 1) * TIME_STEP_MINUTES)

                # Extraer valores de predicción para este paso
                step_predictions = predictions[step]

                # Crear diccionario con valores crudos
                step_data = {
                    "time": f"+{step*TIME_STEP_MINUTES + TIME_STEP_MINUTES}min",
                    "timestamp": step_timestamp.isoformat(),
                    "temp": round(float(step_predictions[0]), 1),  # temperatura
                    "precipitation_mm": max(0, round(float(step_predictions[1]), 2)),  # precipitación
                    "humidity": round(float(step_predictions[2]), 1),  # humedad
                    "wind_speed": round(float(step_predictions[3]), 1),  # velocidad viento
                    "pressure": round(float(step_predictions[4]), 1),  # presión
                    "cloud_cover": round(float(step_predictions[5]), 1)  # nubosidad
                }

                # Generar descripción textual del clima
                step_data["desc"] = self._generate_weather_description(step_data)

                forecast.append(step_data)

            # Resultado final
            result = {
                "postal_code": postal_code,
                "coords": [round(lat, 3), round(lon, 3)],
                "forecast": forecast,
                "generated_at": datetime.utcnow().isoformat(),
                "model_info": {
                    "input_window": "6 hours",
                    "output_window": "6 hours",
                    "time_step": f"{TIME_STEP_MINUTES} minutes",
                    "n_steps": OUTPUT_STEPS
                }
            }

            return result

        except Exception as e:
            logger.error(f"Error formateando resultado: {e}")
            return {
                "error": "Error formateando pronóstico",
                "postal_code": postal_code,
                "coords": [lat, lon]
            }

    def _generate_weather_description(self, step_data: Dict[str, Any]) -> str:
        """
        Genera una descripción textual del estado del tiempo
        Args:
            step_data: Datos del paso temporal
        Returns:
            Descripción textual
        """
        try:
            temp = step_data["temp"]
            rain = step_data["precipitation_mm"]
            clouds = step_data["cloud_cover"]
            wind = step_data["wind_speed"]

            # Descripciones basadas en umbrales
            descriptions = []

            # Temperatura
            if temp < WEATHER_THRESHOLDS["cold"]:
                descriptions.append("Frío")
            elif temp > WEATHER_THRESHOLDS["hot"]:
                descriptions.append("Caluroso")
            else:
                descriptions.append("Temperatura agradable")

            # Precipitación
            if rain > WEATHER_THRESHOLDS["rain_heavy"]:
                descriptions.append("Lluvia torrencial")
            elif rain > WEATHER_THRESHOLDS["rain_moderate"]:
                descriptions.append("Lluvia fuerte")
            elif rain > WEATHER_THRESHOLDS["rain_light"]:
                descriptions.append("Lluvia ligera")
            else:
                # Nubosidad
                if clouds > WEATHER_THRESHOLDS["cloudy"]:
                    descriptions.append("Nublado")
                elif clouds > 25:
                    descriptions.append("Parcialmente nublado")
                else:
                    descriptions.append("Despejado")

            # Viento
            if wind > WEATHER_THRESHOLDS["windy"]:
                descriptions.append("Ventoso")

            return ", ".join(descriptions)

        except Exception as e:
            logger.error(f"Error generando descripción: {e}")
            return "Condiciones variables"

    def display_forecast_console(self, forecast_result: Dict[str, Any]):
        """
        Muestra el pronóstico en formato de consola legible
        Args:
            forecast_result: Resultado de get_weather_forecast()
        """
        try:
            if "error" in forecast_result:
                print(f"ERROR: {forecast_result['error']}")
                return

            print("="*60)
            print(f"Codigo postal: {forecast_result['postal_code']}")
            print(f"Coordenadas: {forecast_result['coords'][0]}, {forecast_result['coords'][1]}")
            print(f"Generado: {forecast_result['generated_at'][:19]} UTC")
            print("="*60)

            print("Pronóstico para las próximas 6 horas:")
            print("-"*60)

            for step in forecast_result['forecast']:
                time = step['time']
                temp = step['temp']
                rain = step['precipitation_mm']
                humidity = step.get('humidity', 0)
                wind_speed = step.get('wind_speed', 0)
                pressure = step.get('pressure', 0)
                desc = step['desc']

                # Formato visual
                rain_icon = "LLUVIA" if rain > 0.1 else "SOLEADO"
                rain_text = f"{rain:.1f}mm" if rain > 0 else "0.0mm"

                print(f"{time:>8} | {temp:>5.1f}C | {rain_text:>7} | {humidity:>4.0f}% | {wind_speed:>4.1f}m/s | {pressure:>6.0f}hPa | {desc}")

            print("-"*60)
            print("El modelo predice cada 15 minutos hasta 6 horas en el futuro")
            print("Variables: temperatura, precipitacion, humedad, viento, presion, nubosidad")

        except Exception as e:
            logger.error(f"Error mostrando pronóstico: {e}")
            print(f"Error mostrando pronostico: {e}")


def get_weather_forecast(postal_code: str) -> Dict[str, Any]:
    """
    Función principal para obtener pronóstico meteorológico
    Args:
        postal_code: Código postal de Chihuahua (5 dígitos)
    Returns:
        Diccionario con pronóstico completo según especificación
    """
    predictor = WeatherPredictor()
    return predictor.get_weather_forecast(postal_code)


# Función de ejemplo para testing
def example_usage():
    """Ejemplo de uso del sistema de predicción"""
    print("Ejemplo de uso del sistema de prediccion meteorologica")
    print("="*60)

    # Crear predictor
    predictor = WeatherPredictor()

    # Ejemplo con código postal de Chihuahua capital
    postal_code = "31000"  # Chihuahua capital

    print(f"Solicitando pronóstico para código postal: {postal_code}")
    print("-"*60)

    # Obtener pronóstico
    forecast = predictor.get_weather_forecast(postal_code)

    # Mostrar resultado
    predictor.display_forecast_console(forecast)

    return forecast


if __name__ == "__main__":
    # Ejecutar ejemplo si se llama directamente
    result = example_usage()

    # También mostrar el resultado raw para debugging
    print("\n" + "="*60)
    print("RESULTADO RAW (primeros 3 pasos):")
    print("="*60)
    if "forecast" in result and result["forecast"]:
        for i, step in enumerate(result["forecast"][:3]):
            print(f"Paso {i+1}: {step}")
    print("="*60)
