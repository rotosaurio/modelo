"""
Módulo para realizar predicciones de precipitación usando modelos entrenados.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import joblib
import requests
import os

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from data_ingest import DataIngestor
from preprocess import DataPreprocessor

logger = logging.getLogger(__name__)


class PrecipitationPredictor:
    """
    Clase para realizar predicciones de precipitación.
    """

    def __init__(self, config: Dict):
        """
        Inicializa el predictor.

        Args:
            config: Configuración del sistema
        """
        self.config = config
        # Usar ruta absoluta basada en el directorio del archivo
        current_dir = Path(__file__).parent.parent
        self.models_dir = current_dir / config['paths']['models']
        self.forecast_horizon = config['data']['forecast_horizon']

        self.model = None
        self.scaler = None
        self.model_metadata = None
        self.feature_columns = None

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Carga el modelo entrenado más reciente.

        Args:
            model_path: Ruta específica al modelo (opcional)

        Returns:
            True si se cargó exitosamente
        """
        try:
            if model_path is None:
                # Buscar el modelo más reciente
                model_files = list(self.models_dir.glob("*_metadata.pkl"))
                if not model_files:
                    logger.error("No se encontraron modelos entrenados")
                    return False

                # Ordenar por fecha de modificación
                model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                metadata_path = model_files[0]
            else:
                metadata_path = Path(model_path)

            # Cargar metadata
            self.model_metadata = joblib.load(metadata_path)
            model_type = self.model_metadata['model_type']

            # Cargar modelo
            model_base = metadata_path.stem.replace('_metadata', '')
            if model_type == 'xgboost':
                model_path = self.models_dir / f"{model_base}.pkl"
                self.model = joblib.load(model_path)
            elif model_type == 'lstm':
                model_path = self.models_dir / f"{model_base}.h5"
                self.model = tf.keras.models.load_model(model_path)

            # Cargar scaler
            scaler_path = self.models_dir / 'scaler.pkl'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            else:
                logger.warning("Scaler no encontrado")

            # Extraer información de features
            self.feature_columns = self.model_metadata.get('feature_columns', [])

            logger.info(f"Modelo cargado: {model_type} desde {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            return False

    def get_recent_data(self, lat: float, lon: float, hours_back: int = 24) -> Optional[pd.DataFrame]:
        """
        Obtiene datos recientes para hacer predicciones.

        Args:
            lat: Latitud
            lon: Longitud
            hours_back: Horas de datos históricos a obtener

        Returns:
            DataFrame con datos recientes
        """
        try:
            # Crear ingestor y preprocessor
            ingestor = DataIngestor(self.config)
            preprocessor = DataPreprocessor(self.config)

            # Calcular fechas
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=hours_back)

            # Intentar descargar datos recientes
            logger.info(f"Descargando datos recientes para ({lat}, {lon})")

            # Priorizar OpenWeatherMap para datos actuales
            openweather_data = ingestor.download_openweather_data(lat, lon)
            if openweather_data:
                # Usar datos de OpenWeather
                df_recent = openweather_data['hourly'].copy()
                df_recent['datetime'] = pd.to_datetime(df_recent['datetime'])
                df_recent = df_recent.set_index('datetime').sort_index()
            else:
                # Fallback: buscar datos existentes
                existing_data = preprocessor.load_raw_data(lat, lon,
                                                         start_date.strftime('%Y-%m-%d'),
                                                         end_date.strftime('%Y-%m-%d'))
                if existing_data:
                    df_recent = preprocessor.align_temporal(existing_data)
                    df_recent = preprocessor.interpolate_spatially_all(df_recent, lat, lon)
                else:
                    logger.warning("No se pudieron obtener datos recientes")
                    return None

            # Filtrar últimas horas
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            df_recent = df_recent[df_recent.index >= cutoff_time]

            if len(df_recent) < 6:
                logger.warning(f"Insuficientes datos recientes: {len(df_recent)} registros")
                return None

            logger.info(f"Datos recientes obtenidos: {len(df_recent)} registros")
            return df_recent

        except Exception as e:
            logger.error(f"Error obteniendo datos recientes: {e}")
            return None

    def prepare_prediction_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Prepara las features para predicción.

        Args:
            df: DataFrame con datos recientes

        Returns:
            DataFrame con features preparadas
        """
        try:
            preprocessor = DataPreprocessor(self.config)

            # Limpiar datos
            df_clean = preprocessor.clean_data(df)

            # Crear features (sin target)
            df_featured = preprocessor.create_features(df_clean)

            # Seleccionar solo las últimas filas disponibles
            df_latest = df_featured.tail(1).copy()

            # Normalizar usando el scaler del modelo
            if self.scaler:
                # Solo normalizar columnas numéricas que no sean target
                numeric_cols = [col for col in self.feature_columns
                              if col in df_latest.columns and
                              df_latest[col].dtype in ['float64', 'int64']]

                if numeric_cols:
                    df_latest[numeric_cols] = self.scaler.transform(df_latest[numeric_cols])

            # Seleccionar solo las features que el modelo espera
            available_features = [col for col in self.feature_columns if col in df_latest.columns]

            if not available_features:
                logger.error("No se encontraron features compatibles con el modelo")
                return None

            X_pred = df_latest[available_features].copy()

            logger.info(f"Features preparadas para predicción: {len(available_features)} features")
            return X_pred

        except Exception as e:
            logger.error(f"Error preparando features: {e}")
            return None

    def predict_rain_probability(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """
        Predice la probabilidad de lluvia para las próximas horas usando datos en tiempo real.

        Args:
            lat: Latitud
            lon: Longitud

        Returns:
            Diccionario con resultados de predicción basados en datos actuales
        """
        try:
            # Obtener datos actuales de OpenWeatherMap
            owm_data = self.get_recent_owm_data(lat, lon)
            if owm_data is None:
                return self.create_fallback_prediction(lat, lon)

            # Analizar condiciones actuales para estimar probabilidad de lluvia
            rain_probability = self.estimate_rain_probability(owm_data)
            will_rain = rain_probability > 0.3  # Umbral más bajo para ser conservador

            # Resultados basados en datos actuales
            result = {
                'rain_probability': round(rain_probability, 3),
                'will_rain_next_6h': will_rain,
                'prediction_time': datetime.now().isoformat(),
                'location': {'lat': lat, 'lon': lon},
                'model_info': {
                    'type': 'real_time_estimator',
                    'description': 'Estimación basada en condiciones actuales y pronóstico',
                    'data_sources': ['openweather']
                },
                'current_conditions': owm_data.get('current', {}),
                'hourly_forecast': owm_data.get('hourly', [])[:6],  # Próximas 6 horas
                'confidence_level': self._calculate_confidence(rain_probability),
                'note': 'Esta es una estimación en tiempo real basada en datos actuales, no un modelo ML entrenado.'
            }

            logger.info(f"Predicción completada: {rain_probability:.3f} probabilidad de lluvia para ({lat}, {lon})")
            return result

        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return self.create_fallback_prediction(lat, lon)

    def get_recent_owm_data(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Obtiene datos actuales y forecast de OpenWeatherMap.

        Args:
            lat: Latitud
            lon: Longitud

        Returns:
            Datos de OpenWeatherMap o None si falla
        """
        api_key = os.getenv('OPENWEATHER_API_KEY')
        if not api_key:
            logger.error("OpenWeather API key no configurada")
            return None

        try:
            # API One Call 3.0
            url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&appid={api_key}&units=metric&exclude=minutely,daily,alerts"

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()
            return {
                'current': data.get('current', {}),
                'hourly': data.get('hourly', [])[:24]  # Próximas 24 horas
            }

        except requests.RequestException as e:
            logger.error(f"Error en solicitud OpenWeather: {e}")
            return None
        except Exception as e:
            logger.error(f"Error procesando datos OpenWeather: {e}")
            return None

    def estimate_rain_probability(self, owm_data: Dict) -> float:
        """
        Estima la probabilidad de lluvia basada en datos de OpenWeatherMap.

        Args:
            owm_data: Datos de OpenWeatherMap

        Returns:
            Probabilidad de lluvia (0-1)
        """
        try:
            current = owm_data.get('current', {})
            hourly = owm_data.get('hourly', [])

            # Factores que influyen en la probabilidad de lluvia
            probability = 0.0

            # 1. Condiciones actuales
            if 'rain' in current and current['rain']:
                probability += 0.4  # Está lloviendo ahora
            if current.get('humidity', 0) > 80:
                probability += 0.2  # Alta humedad
            if current.get('clouds', 0) > 70:
                probability += 0.2  # Muy nublado

            # 2. Pronóstico de las próximas horas
            rain_hours = 0
            pop_sum = 0

            for hour in hourly[:6]:  # Próximas 6 horas
                pop = hour.get('pop', 0)  # Probability of precipitation
                pop_sum += pop

                if pop > 0.3:  # Más del 30% de probabilidad
                    rain_hours += 1

                # Si hay lluvia pronosticada
                if 'rain' in hour and hour['rain']:
                    probability += 0.1

            # Promedio de probabilidad de precipitación
            avg_pop = pop_sum / len(hourly[:6]) if hourly else 0
            probability += avg_pop * 0.3

            # Bonus por horas consecutivas con lluvia probable
            if rain_hours >= 3:
                probability += 0.2

            # Asegurar que esté entre 0 y 1
            probability = max(0.0, min(1.0, probability))

            return probability

        except Exception as e:
            logger.error(f"Error estimando probabilidad de lluvia: {e}")
            return 0.1  # Valor conservador por defecto

    def create_fallback_prediction(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Crea una predicción por defecto cuando no hay datos disponibles.

        Args:
            lat: Latitud
            lon: Longitud

        Returns:
            Predicción por defecto
        """
        return {
            'rain_probability': 0.1,  # Baja probabilidad por defecto
            'will_rain_next_6h': False,
            'prediction_time': datetime.now().isoformat(),
            'location': {'lat': lat, 'lon': lon},
            'model_info': {
                'type': 'fallback',
                'description': 'Predicción por defecto - no hay datos disponibles'
            },
            'confidence_level': 'low',
            'note': 'No se pudieron obtener datos meteorológicos. Esta es una estimación conservadora.'
        }

    def _calculate_confidence(self, probability: float) -> str:
        """
        Calcula el nivel de confianza de la predicción.

        Args:
            probability: Probabilidad de lluvia

        Returns:
            Nivel de confianza como string
        """
        if probability < 0.3:
            return "low"
        elif probability < 0.7:
            return "medium"
        else:
            return "high"

    def batch_predict(self, locations: List[Tuple[float, float]]) -> List[Optional[Dict[str, Any]]]:
        """
        Realiza predicciones para múltiples ubicaciones.

        Args:
            locations: Lista de tuplas (lat, lon)

        Returns:
            Lista de resultados de predicción
        """
        results = []
        for lat, lon in locations:
            try:
                result = self.predict_rain_probability(lat, lon)
                results.append(result)
            except Exception as e:
                logger.error(f"Error prediciendo para ({lat}, {lon}): {e}")
                results.append(None)

        return results


# Funciones de conveniencia
def predict_rain_for_location(lat: float, lon: float, config: Dict,
                            model_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Función de conveniencia para predecir lluvia en una ubicación usando datos en tiempo real.

    Args:
        lat: Latitud
        lon: Longitud
        config: Configuración
        model_path: Ruta al modelo (opcional, ignorado en modo tiempo real)

    Returns:
        Resultado de la predicción basado en datos actuales
    """
    predictor = PrecipitationPredictor(config)
    # En modo tiempo real, no necesitamos cargar modelo pre-entrenado
    return predictor.predict_rain_probability(lat, lon)


def create_mock_prediction(lat: float, lon: float) -> Dict[str, Any]:
    """
    Crea una predicción simulada para testing.

    Args:
        lat: Latitud
        lon: Longitud

    Returns:
        Predicción simulada
    """
    probability = np.random.random()
    return {
        'rain_probability': round(probability, 3),
        'will_rain_next_6h': probability > 0.5,
        'prediction_time': datetime.now().isoformat(),
        'location': {'lat': lat, 'lon': lon},
        'model_info': {
            'type': 'mock_model',
            'trained_at': '2024-01-01',
            'auc_score': 0.75
        },
        'data_sources': ['mock_data'],
        'confidence_level': 'medium',
        'note': 'Esta es una predicción simulada para testing'
    }


if __name__ == "__main__":
    # Ejemplo de uso
    from utils import load_config, setup_logging

    setup_logging()
    config = load_config("../config.yaml")

    # Coordenadas de Chihuahua
    lat, lon = 28.6333, -106.0691

    try:
        result = predict_rain_for_location(lat, lon, config)

        if result:
            print("=== PREDICCIÓN DE LLUVIA ===")
            print(f"Ubicación: {result['location']['lat']}, {result['location']['lon']}")
            print(f"Probabilidad de lluvia: {result['rain_probability']:.1%}")
            print(f"¿Lloverá en 6 horas?: {'Sí' if result['will_rain_next_6h'] else 'No'}")
            print(f"Confianza: {result['confidence_level']}")
            print(f"Modelo: {result['model_info']['type']} (AUC: {result['model_info']['auc_score']})")
        else:
            print("No se pudo realizar la predicción")
            print("Usando predicción simulada...")

            mock_result = create_mock_prediction(lat, lon)
            print(f"Predicción simulada: {mock_result['rain_probability']:.1%} probabilidad")

    except Exception as e:
        print(f"Error en predicción: {e}")
