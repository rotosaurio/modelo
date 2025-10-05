"""
Módulo de preprocesamiento de datos climáticos
Unifica datos de múltiples fuentes (IMERG, ERA5, Meteostat) en intervalos de 15 minutos
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config import WEATHER_VARIABLES, TIME_STEP_MINUTES, SYNTHETIC_CONFIG
from data_imerg import IMERGDataDownloader
from data_era5 import ERA5DataDownloader
from data_meteostat import MeteostatDataDownloader
from data_synthetic import SyntheticWeatherGenerator

logger = logging.getLogger(__name__)


class WeatherDataPreprocessor:
    """Clase para preprocesar y unificar datos climáticos"""

    def __init__(self):
        self.imerg_downloader = IMERGDataDownloader()
        self.era5_downloader = ERA5DataDownloader()
        self.meteostat_downloader = MeteostatDataDownloader()
        self.synthetic_generator = SyntheticWeatherGenerator()

        # Escaladores para normalización
        self.scaler = None
        self.feature_columns = []

    def collect_weather_data(self, lat: float, lon: float, postal_code: str,
                           start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Recopila datos climáticos de todas las fuentes para un período específico
        Args:
            lat: Latitud
            lon: Longitud
            postal_code: Código postal
            start_date: Fecha de inicio
            end_date: Fecha de fin
        Returns:
            DataFrame unificado con todos los datos climáticos
        """
        try:
            logger.info(f"Recopilando datos climáticos para {postal_code} del {start_date.date()} al {end_date.date()}")

            # Si está forzado el uso de datos sintéticos, usarlos directamente
            if SYNTHETIC_CONFIG.get("force_synthetic", False):
                logger.info("Usando datos sintéticos forzados (modo desarrollo)")
                synthetic_data = self.synthetic_generator.generate_synthetic_weather(
                    lat, lon, start_date, end_date
                )
                if not synthetic_data.empty:
                    logger.info(f"Datos sintéticos generados: {len(synthetic_data)} registros")
                    return synthetic_data
                else:
                    logger.error("No se pudieron generar datos sintéticos")
                    return None

            # Recopilar datos de cada fuente
            data_sources = {}

            # 1. Datos IMERG (precipitación)
            logger.info("Obteniendo datos IMERG...")
            imerg_data = self.imerg_downloader.download_imerg_data(lat, lon, start_date, end_date)
            if imerg_data is not None:
                data_sources['imerg'] = imerg_data
                logger.info(f"IMERG: {len(imerg_data)} registros")
            else:
                logger.warning("No se pudieron obtener datos IMERG")

            # 2. Datos ERA5 (variables atmosféricas)
            logger.info("Obteniendo datos ERA5...")
            era5_data = self.era5_downloader.download_era5_data(lat, lon, start_date, end_date)
            if era5_data is not None:
                era5_df = self.era5_downloader.process_era5_data(era5_data, lat, lon)
                if not era5_df.empty:
                    data_sources['era5'] = era5_df
                    logger.info(f"ERA5: {len(era5_df)} registros")
            else:
                logger.warning("No se pudieron obtener datos ERA5")

            # 3. Datos de estaciones locales (Meteostat)
            logger.info("Obteniendo datos de estaciones locales...")
            station_data = self.meteostat_downloader.get_combined_station_data(lat, lon, start_date, end_date)
            if station_data is not None and not station_data.empty:
                data_sources['meteostat'] = station_data
                logger.info(f"Meteostat: {len(station_data)} registros")
            else:
                logger.warning("No se pudieron obtener datos de estaciones locales")

            if not data_sources:
                logger.warning("No se pudieron obtener datos de ninguna API, usando datos sintéticos...")
                # Generar datos sintéticos como fallback
                synthetic_data = self.synthetic_generator.generate_synthetic_weather(
                    lat, lon, start_date, end_date
                )
                if not synthetic_data.empty:
                    logger.info(f"Datos sintéticos generados: {len(synthetic_data)} registros")
                    return synthetic_data
                else:
                    logger.error("No se pudieron generar datos sintéticos")
                    return None

            # Unificar todos los datos
            unified_data = self._unify_data_sources(data_sources)

            if unified_data.empty:
                logger.warning("Los datos unificados están vacíos, usando datos sintéticos...")
                # Fallback a datos sintéticos
                synthetic_data = self.synthetic_generator.generate_synthetic_weather(
                    lat, lon, start_date, end_date
                )
                if not synthetic_data.empty:
                    logger.info(f"Datos sintéticos generados: {len(synthetic_data)} registros")
                    return synthetic_data
                else:
                    logger.error("No se pudieron generar datos sintéticos")
                    return None

            logger.info(f"Datos unificados: {len(unified_data)} registros, {len(unified_data.columns)} variables")
            return unified_data

        except Exception as e:
            logger.error(f"Error recopilando datos climáticos: {e}")
            return None

    def _unify_data_sources(self, data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Unifica datos de múltiples fuentes en un solo DataFrame
        Args:
            data_sources: Diccionario con DataFrames de cada fuente
        Returns:
            DataFrame unificado
        """
        try:
            # Crear índice común de 15 minutos
            all_timestamps = set()
            for source_name, df in data_sources.items():
                all_timestamps.update(df.index)

            if not all_timestamps:
                return pd.DataFrame()

            start_time = min(all_timestamps)
            end_time = max(all_timestamps)

            # Crear índice de 15 minutos
            common_index = pd.date_range(start=start_time, end=end_time, freq='15min')

            # DataFrame vacío con índice común
            unified_df = pd.DataFrame(index=common_index)

            # Variables disponibles por fuente
            source_variables = {
                'imerg': ['precipitation_mm'],
                'era5': ['temperature_celsius', 'relative_humidity_percent', 'wind_speed_ms',
                        'wind_direction_deg', 'pressure_hpa', 'cloud_cover_percent'],
                'meteostat': ['temperature_celsius', 'dewpoint_celsius', 'relative_humidity_percent',
                             'precipitation_mm', 'wind_speed_ms', 'wind_direction_deg',
                             'pressure_hpa', 'wind_gust_ms']
            }

            # Unificar cada variable
            for source_name, df in data_sources.items():
                available_vars = source_variables.get(source_name, [])

                for var in available_vars:
                    if var in df.columns:
                        # Reindexar a 15 minutos
                        var_series = df[var].reindex(common_index)

                        # Interpolar valores faltantes
                        var_series = var_series.interpolate(method='linear', limit=4)  # máximo 1 hora

                        # Usar prefijo para evitar conflictos
                        unified_df[f"{source_name}_{var}"] = var_series

            # Crear variables finales combinando fuentes (con prioridad)
            self._create_final_variables(unified_df)

            # Eliminar columnas intermedias, mantener solo las finales
            final_columns = list(WEATHER_VARIABLES.values())
            existing_final_columns = [col for col in final_columns if col in unified_df.columns]
            unified_df = unified_df[existing_final_columns]

            # Rellenar valores faltantes restantes con forward/backward fill
            unified_df = unified_df.ffill().bfill()

            return unified_df

        except Exception as e:
            logger.error(f"Error unificando fuentes de datos: {e}")
            return pd.DataFrame()

    def _create_final_variables(self, df: pd.DataFrame):
        """
        Crea variables finales combinando datos de múltiples fuentes con prioridades
        Args:
            df: DataFrame con variables de todas las fuentes
        """
        try:
            # Temperatura: prioridad Meteostat > ERA5
            if 'meteostat_temperature_celsius' in df.columns:
                df[WEATHER_VARIABLES['temperature']] = df['meteostat_temperature_celsius']
            elif 'era5_temperature_celsius' in df.columns:
                df[WEATHER_VARIABLES['temperature']] = df['era5_temperature_celsius']

            # Humedad: prioridad Meteostat > ERA5
            if 'meteostat_relative_humidity_percent' in df.columns:
                df[WEATHER_VARIABLES['humidity']] = df['meteostat_relative_humidity_percent']
            elif 'era5_relative_humidity_percent' in df.columns:
                df[WEATHER_VARIABLES['humidity']] = df['era5_relative_humidity_percent']

            # Viento: combinar velocidad y dirección
            # Velocidad: prioridad Meteostat > ERA5
            if 'meteostat_wind_speed_ms' in df.columns:
                df[WEATHER_VARIABLES['wind_speed']] = df['meteostat_wind_speed_ms']
            elif 'era5_wind_speed_ms' in df.columns:
                df[WEATHER_VARIABLES['wind_speed']] = df['era5_wind_speed_ms']

            # Dirección: prioridad Meteostat > ERA5
            if 'meteostat_wind_direction_deg' in df.columns:
                df[WEATHER_VARIABLES['wind_direction']] = df['meteostat_wind_direction_deg']
            elif 'era5_wind_direction_deg' in df.columns:
                df[WEATHER_VARIABLES['wind_direction']] = df['era5_wind_direction_deg']

            # Presión: prioridad Meteostat > ERA5
            if 'meteostat_pressure_hpa' in df.columns:
                df[WEATHER_VARIABLES['pressure']] = df['meteostat_pressure_hpa']
            elif 'era5_pressure_hpa' in df.columns:
                df[WEATHER_VARIABLES['pressure']] = df['era5_pressure_hpa']

            # Precipitación: prioridad IMERG > Meteostat
            if 'imerg_precipitation_mm' in df.columns:
                df[WEATHER_VARIABLES['precipitation']] = df['imerg_precipitation_mm']
            elif 'meteostat_precipitation_mm' in df.columns:
                df[WEATHER_VARIABLES['precipitation']] = df['meteostat_precipitation_mm']

            # Nubosidad: solo ERA5
            if 'era5_cloud_cover_percent' in df.columns:
                df[WEATHER_VARIABLES['cloud_cover']] = df['era5_cloud_cover_percent']

        except Exception as e:
            logger.error(f"Error creando variables finales: {e}")

    def collect_prediction_data(self, lat: float, lon: float, postal_code: str) -> Optional[pd.DataFrame]:
        """
        Recopila datos para hacer predicciones usando fechas históricas (septiembre 2024)
        Args:
            lat: Latitud
            lon: Longitud
            postal_code: Código postal
        Returns:
            DataFrame con datos históricos de 6 horas
        """
        try:
            logger.info(f"Recopilando datos históricos para predicción en {postal_code}")

            # Usar fechas históricas fijas (septiembre 2024) para evitar problemas de datos futuros
            base_date = pd.Timestamp('2024-09-15 12:00:00')
            end_date = base_date + pd.DateOffset(hours=6)
            start_date = base_date

            logger.info(f"Usando fechas históricas: {start_date} a {end_date}")

            # Intentar obtener datos históricos de cada fuente
            data_sources = {}

            # 1. Datos ERA5 históricos
            try:
                era5_data = self.era5_downloader.get_data(
                    lat, lon, start_date.to_pydatetime(), end_date.to_pydatetime()
                )
                if not era5_data.empty:
                    data_sources['era5'] = era5_data
                    logger.info(f"Datos ERA5 históricos obtenidos: {len(era5_data)} registros")
            except Exception as e:
                logger.warning(f"Error obteniendo datos ERA5 históricos: {e}")

            # 2. Datos IMERG históricos
            try:
                imerg_data = self.imerg_downloader.get_data(
                    lat, lon, start_date.to_pydatetime(), end_date.to_pydatetime()
                )
                if not imerg_data.empty:
                    data_sources['imerg'] = imerg_data
                    logger.info(f"Datos IMERG históricos obtenidos: {len(imerg_data)} registros")
            except Exception as e:
                logger.warning(f"Error obteniendo datos IMERG históricos: {e}")

            # 3. Datos de estaciones históricos
            try:
                station_data = self.meteostat_downloader.get_data(
                    lat, lon, start_date.to_pydatetime(), end_date.to_pydatetime()
                )
                if not station_data.empty:
                    data_sources['meteostat'] = station_data
                    logger.info(f"Datos de estaciones históricos obtenidos: {len(station_data)} registros")
            except Exception as e:
                logger.warning(f"Error obteniendo datos de estaciones históricos: {e}")

            # Si no hay datos reales disponibles, usar datos sintéticos como fallback
            if not data_sources:
                logger.warning("No se pudieron obtener datos históricos de ninguna fuente, usando datos sintéticos")
                synthetic_data = self.synthetic_generator.generate_synthetic_weather(
                    lat, lon, start_date.to_pydatetime(), end_date.to_pydatetime()
                )

                if not synthetic_data.empty:
                    logger.info(f"Datos sintéticos generados para predicción: {len(synthetic_data)} registros")
                    return synthetic_data
                else:
                    logger.error("No se pudieron generar datos sintéticos para predicción")
                    return None

            # Unificar datos recientes
            unified_data = self._unify_data_sources(data_sources)

            if unified_data.empty or len(unified_data) < 20:  # Necesitamos al menos 5 horas de datos
                logger.warning("Datos unificados insuficientes, usando datos sintéticos como fallback")
                synthetic_data = self.synthetic_generator.generate_synthetic_weather(
                    lat, lon, start_date.to_pydatetime(), end_date.to_pydatetime()
                )

                if not synthetic_data.empty:
                    logger.info(f"Datos sintéticos generados para predicción: {len(synthetic_data)} registros")
                    return synthetic_data
                else:
                    logger.error("No se pudieron generar datos sintéticos para predicción")
                    return None

            logger.info(f"Datos para predicción recopilados: {len(unified_data)} registros")
            return unified_data

        except Exception as e:
            logger.error(f"Error recopilando datos para predicción: {e}")
            # Fallback final a datos sintéticos
            try:
                base_date = pd.Timestamp('2024-09-15 12:00:00')
                end_date = base_date + pd.DateOffset(hours=6)
                start_date = base_date
                
                synthetic_data = self.synthetic_generator.generate_synthetic_weather(
                    lat, lon, start_date.to_pydatetime(), end_date.to_pydatetime()
                )
                logger.info(f"Fallback final - Datos sintéticos generados: {len(synthetic_data)} registros")
                return synthetic_data
            except Exception as fallback_error:
                logger.error(f"Error en fallback final: {fallback_error}")
                return None

    def normalize_data(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Normaliza las variables numéricas del DataFrame con manejo robusto de NaN e infinitos
        Args:
            df: DataFrame con datos a normalizar
            fit_scaler: Si True, ajusta el escalador; si False, usa el escalador existente
        Returns:
            DataFrame normalizado
        """
        try:
            df_normalized = df.copy()

            # Variables a normalizar (excluir categóricas si las hay)
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

            # Manejo robusto de valores problemáticos
            for col in numeric_columns:
                # Reemplazar infinitos con NaN
                df_normalized[col] = df_normalized[col].replace([np.inf, -np.inf], np.nan)
                # Imputar NaN con la mediana de la columna
                median_val = df_normalized[col].median()
                if pd.isna(median_val):
                    median_val = 0  # Si toda la columna es NaN, usar 0
                df_normalized[col] = df_normalized[col].fillna(median_val)

            if fit_scaler or self.scaler is None:
                self.scaler = StandardScaler()
                self.scaler.fit(df_normalized[numeric_columns])
                self.feature_columns = numeric_columns
                logger.info(f"Escalador ajustado con {len(numeric_columns)} variables")
            else:
                # Verificar que las columnas coincidan
                if set(numeric_columns) != set(self.feature_columns):
                    logger.warning("Las columnas no coinciden con el escalador entrenado")
                    # Reajustar escalador
                    self.scaler = StandardScaler()
                    self.scaler.fit(df_normalized[numeric_columns])
                    self.feature_columns = numeric_columns

            # Normalizar
            normalized_values = self.scaler.transform(df_normalized[numeric_columns])
            df_normalized[numeric_columns] = normalized_values

            logger.debug(f"Datos normalizados: {df_normalized.shape}")
            return df_normalized

        except Exception as e:
            logger.error(f"Error normalizando datos: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return df

    def denormalize_data(self, df_normalized: pd.DataFrame) -> pd.DataFrame:
        """
        Desnormaliza datos previamente normalizados
        Args:
            df_normalized: DataFrame normalizado
        Returns:
            DataFrame desnormalizado
        """
        try:
            if self.scaler is None:
                logger.error("No hay escalador disponible para desnormalizar")
                return df_normalized

            df_denormalized = df_normalized.copy()

            # Variables numéricas que fueron normalizadas
            numeric_columns = [col for col in self.feature_columns if col in df_denormalized.columns]

            # Desnormalizar
            denormalized_values = self.scaler.inverse_transform(df_denormalized[numeric_columns])
            df_denormalized[numeric_columns] = denormalized_values

            return df_denormalized

        except Exception as e:
            logger.error(f"Error desnormalizando datos: {e}")
            return df_normalized

    def prepare_sequences(self, df: pd.DataFrame, input_steps: int, output_steps: int,
                         target_variables: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara secuencias de entrada y salida para el modelo
        Args:
            df: DataFrame con datos históricos
            input_steps: Número de pasos de entrada (6 horas = 24 pasos de 15 min)
            output_steps: Número de pasos de salida (6 horas = 24 pasos de 15 min)
            target_variables: Variables a predecir
        Returns:
            Tupla (X, y) con secuencias de entrada y salida
        """
        try:
            # Asegurar que tenemos suficientes datos
            min_length = input_steps + output_steps
            if len(df) < min_length:
                logger.error(f"Datos insuficientes: {len(df)} < {min_length}")
                return np.array([]), np.array([])

            # Limpiar y preparar datos para numpy
            logger.info("Limpiando datos para conversión a tensores...")

            # Seleccionar solo columnas numéricas
            numeric_df = df.select_dtypes(include=[np.number])

            # Rellenar NaN con valores apropiados
            numeric_df = numeric_df.ffill().bfill().fillna(0)

            # Asegurar que todos los valores sean float32
            numeric_df = numeric_df.astype(np.float32)

            # Obtener arrays de numpy
            data_array = numeric_df.values
            n_samples = len(numeric_df) - input_steps - output_steps + 1

            logger.info(f"Datos preparados: {data_array.shape}, tipo: {data_array.dtype}")

            X = []
            y = []

            for i in range(n_samples):
                # Secuencia de entrada
                input_seq = data_array[i:i + input_steps]
                X.append(input_seq)

                # Secuencia de salida (solo variables objetivo)
                target_indices = [numeric_df.columns.get_loc(col) for col in target_variables if col in numeric_df.columns]
                if target_indices:
                    output_seq = data_array[i + input_steps:i + input_steps + output_steps, target_indices]
                    y.append(output_seq)

            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)

            # Validar que los arrays no estén vacíos
            if len(X) == 0 or len(y) == 0:
                logger.error("No se pudieron crear secuencias válidas")
                return np.array([]), np.array([])

            logger.info(f"Secuencias preparadas: {len(X)} muestras, entrada {X.shape}, salida {y.shape}")
            return X, y

        except Exception as e:
            logger.error(f"Error preparando secuencias: {e}")
            return np.array([]), np.array([])
