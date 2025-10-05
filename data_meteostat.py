"""
Módulo para descargar datos de estaciones meteorológicas locales usando Meteostat
Proporciona observaciones reales de temperatura, precipitación y otras variables
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from meteostat import Point, Hourly, Stations

from config import WEATHER_DATA_DIR

logger = logging.getLogger(__name__)


class MeteostatDataDownloader:
    """Clase para descargar datos de estaciones meteorológicas con Meteostat"""

    def __init__(self):
        self.data_dir = WEATHER_DATA_DIR / "meteostat"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def find_nearest_stations(self, lat: float, lon: float, radius_km: float = 50,
                            limit: int = 5) -> List[dict]:
        """
        Encuentra estaciones meteorológicas más cercanas a coordenadas específicas
        Args:
            lat: Latitud
            lon: Longitud
            radius_km: Radio de búsqueda en kilómetros
            limit: Número máximo de estaciones a retornar
        Returns:
            Lista de diccionarios con información de estaciones
        """
        try:
            logger.info(f"Buscando estaciones cercanas a ({lat:.3f}, {lon:.3f}) dentro de {radius_km}km")

            # Crear punto de referencia
            point = Point(lat, lon)

            # Buscar estaciones dentro del radio
            stations = Stations()
            stations = stations.nearby(lat, lon, radius_km)

            # Obtener dataframe de estaciones
            df_stations = stations.fetch(limit)

            if df_stations.empty:
                logger.warning("No se encontraron estaciones meteorológicas cercanas")
                return []

            # Convertir a lista de diccionarios
            station_list = []
            for idx, row in df_stations.iterrows():
                station_info = {
                    'id': idx,
                    'name': row.get('name', 'Unknown'),
                    'country': row.get('country', ''),
                    'region': row.get('region', ''),
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'elevation': row.get('elevation', 0),
                    'distance_km': point.distance(Point(row['latitude'], row['longitude'])) / 1000
                }
                station_list.append(station_info)

            logger.info(f"Encontradas {len(station_list)} estaciones cercanas")
            return station_list

        except Exception as e:
            logger.error(f"Error buscando estaciones cercanas: {e}")
            return []

    def download_station_data(self, station_id: str, start_date: datetime,
                            end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Descarga datos horarios de una estación específica
        Args:
            station_id: ID de la estación
            start_date: Fecha de inicio
            end_date: Fecha de fin
        Returns:
            DataFrame con datos de la estación o None si falla
        """
        try:
            logger.debug(f"Descargando datos de estación {station_id} del {start_date.date()} al {end_date.date()}")

            # Obtener datos horarios
            data = Hourly(station_id, start_date, end_date)
            df = data.fetch()

            if df.empty:
                logger.warning(f"No hay datos disponibles para estación {station_id}")
                return None

            # Renombrar columnas para consistencia
            column_mapping = {
                'temp': 'temperature_celsius',
                'dwpt': 'dewpoint_celsius',
                'rhum': 'relative_humidity_percent',
                'prcp': 'precipitation_mm',
                'snow': 'snow_mm',
                'wdir': 'wind_direction_deg',
                'wspd': 'wind_speed_ms',  # Meteostat da velocidad en km/h, convertir a m/s
                'wpgt': 'wind_gust_ms',
                'pres': 'pressure_hpa',
                'tsun': 'sunshine_minutes',
                'coco': 'condition_code'
            }

            df = df.rename(columns=column_mapping)

            # Convertir unidades
            if 'wind_speed_ms' in df.columns:
                df['wind_speed_ms'] = df['wind_speed_ms'] * 1000 / 3600  # km/h to m/s

            if 'wind_gust_ms' in df.columns:
                df['wind_gust_ms'] = df['wind_gust_ms'] * 1000 / 3600  # km/h to m/s

            # Agregar columna de ID de estación
            df['station_id'] = station_id

            logger.debug(f"Descargados {len(df)} registros de estación {station_id}")
            return df

        except Exception as e:
            logger.error(f"Error descargando datos de estación {station_id}: {e}")
            return None

    def get_combined_station_data(self, lat: float, lon: float, start_date: datetime,
                                end_date: datetime, max_stations: int = 3) -> Optional[pd.DataFrame]:
        """
        Obtiene datos combinados de múltiples estaciones cercanas
        Args:
            lat: Latitud
            lon: Longitud
            start_date: Fecha de inicio
            end_date: Fecha de fin
            max_stations: Número máximo de estaciones a usar
        Returns:
            DataFrame con datos combinados o None si falla
        """
        try:
            # Encontrar estaciones cercanas
            stations = self.find_nearest_stations(lat, lon, radius_km=100, limit=max_stations)

            if not stations:
                logger.warning("No se encontraron estaciones para combinar datos")
                return None

            all_data = []

            for station in stations:
                station_id = station['id']
                df_station = self.download_station_data(station_id, start_date, end_date)

                if df_station is not None and not df_station.empty:
                    # Agregar información de distancia y peso
                    df_station['distance_km'] = station['distance_km']
                    df_station['weight'] = 1 / (1 + station['distance_km'])  # Peso inversamente proporcional a distancia

                    all_data.append(df_station)

            if not all_data:
                logger.warning("No se pudieron descargar datos de ninguna estación")
                return None

            # Combinar datos de todas las estaciones
            combined_df = self._merge_station_data(all_data)

            logger.info(f"Datos combinados de {len(all_data)} estaciones: {len(combined_df)} registros")
            return combined_df

        except Exception as e:
            logger.error(f"Error obteniendo datos combinados de estaciones: {e}")
            return None

    def _merge_station_data(self, station_dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Combina datos de múltiples estaciones usando promedio ponderado por distancia
        Args:
            station_dataframes: Lista de DataFrames de estaciones
        Returns:
            DataFrame combinado
        """
        try:
            if not station_dataframes:
                return pd.DataFrame()

            # Obtener todos los timestamps únicos
            all_timestamps = set()
            for df in station_dataframes:
                all_timestamps.update(df.index)

            all_timestamps = sorted(list(all_timestamps))

            # Variables numéricas para promediar
            numeric_vars = [
                'temperature_celsius', 'dewpoint_celsius', 'relative_humidity_percent',
                'precipitation_mm', 'wind_speed_ms', 'wind_direction_deg',
                'pressure_hpa', 'wind_gust_ms'
            ]

            # Crear DataFrame vacío con todos los timestamps
            combined_data = []

            for ts in all_timestamps:
                ts_data = {'timestamp': ts}

                # Recopilar datos de todas las estaciones para este timestamp
                station_values = {}
                total_weight = 0

                for df in station_dataframes:
                    if ts in df.index:
                        row = df.loc[ts]
                        weight = row.get('weight', 1.0)
                        total_weight += weight

                        for var in numeric_vars:
                            if var in row.index and pd.notna(row[var]):
                                if var not in station_values:
                                    station_values[var] = []
                                station_values[var].append((row[var], weight))

                # Calcular promedio ponderado para cada variable
                for var in numeric_vars:
                    if var in station_values and total_weight > 0:
                        weighted_sum = sum(val * weight for val, weight in station_values[var])
                        ts_data[var] = weighted_sum / total_weight

                combined_data.append(ts_data)

            # Crear DataFrame final
            df_combined = pd.DataFrame(combined_data)
            df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
            df_combined = df_combined.set_index('timestamp').sort_index()

            # Interpolar valores faltantes (máximo 1 hora de gap)
            df_combined = df_combined.interpolate(method='linear', limit=4)  # 4 * 15min = 1 hora

            logger.debug(f"Datos de estaciones combinados: {len(df_combined)} registros")
            return df_combined

        except Exception as e:
            logger.error(f"Error combinando datos de estaciones: {e}")
            return pd.DataFrame()

    def get_recent_station_data(self, lat: float, lon: float, hours: int = 24) -> Optional[pd.DataFrame]:
        """
        Descarga datos recientes de estaciones para predicción
        Args:
            lat: Latitud
            lon: Longitud
            hours: Número de horas hacia atrás
        Returns:
            DataFrame con datos recientes
        """
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(hours=hours)

            return self.get_combined_station_data(lat, lon, start_date, end_date)

        except Exception as e:
            logger.error(f"Error obteniendo datos recientes de estaciones: {e}")
            return None

    def _interpolate_to_15min(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpola datos horarios a 15 minutos
        Args:
            df: DataFrame con datos horarios
        Returns:
            DataFrame con datos de 15 minutos
        """
        try:
            # Reindexar a 15 minutos
            df_15min = df.resample('15min').interpolate(method='linear')

            # Aplicar límites físicos
            if 'temperature_celsius' in df_15min.columns:
                df_15min['temperature_celsius'] = df_15min['temperature_celsius'].clip(-50, 60)

            if 'relative_humidity_percent' in df_15min.columns:
                df_15min['relative_humidity_percent'] = df_15min['relative_humidity_percent'].clip(0, 100)

            if 'precipitation_mm' in df_15min.columns:
                df_15min['precipitation_mm'] = df_15min['precipitation_mm'].clip(0, 500)

            if 'wind_speed_ms' in df_15min.columns:
                df_15min['wind_speed_ms'] = df_15min['wind_speed_ms'].clip(0, 100)

            if 'pressure_hpa' in df_15min.columns:
                df_15min['pressure_hpa'] = df_15min['pressure_hpa'].clip(800, 1100)

            logger.info(f"Datos interpolados de {len(df)} a {len(df_15min)} registros (15min)")
            return df_15min

        except Exception as e:
            logger.error(f"Error interpolando datos de estaciones: {e}")
            return df

    def save_station_data(self, df: pd.DataFrame, postal_code: str, start_date: datetime) -> Path:
        """
        Guarda datos de estaciones en archivo
        Args:
            df: DataFrame con datos
            postal_code: Código postal
            start_date: Fecha de inicio
        Returns:
            Path al archivo guardado
        """
        filename = f"meteostat_{postal_code}_{start_date.strftime('%Y%m%d_%H%M')}.csv"
        filepath = self.data_dir / filename

        df.to_csv(filepath)
        logger.info(f"Datos de estaciones guardados: {filepath}")

        return filepath

    def load_station_data(self, postal_code: str, start_date: datetime) -> Optional[pd.DataFrame]:
        """
        Carga datos de estaciones desde archivo
        Args:
            postal_code: Código postal
            start_date: Fecha de inicio
        Returns:
            DataFrame con datos o None si no existe
        """
        pattern = f"meteostat_{postal_code}_{start_date.strftime('%Y%m%d_%H%M')}*.csv"
        files = list(self.data_dir.glob(pattern))

        if not files:
            return None

        try:
            df = pd.read_csv(files[0], index_col='timestamp', parse_dates=True)
            logger.info(f"Datos de estaciones cargados: {files[0]}")
            return df
        except Exception as e:
            logger.error(f"Error cargando datos de estaciones: {e}")
            return None

    def get_station_data_for_prediction(self, lat: float, lon: float, postal_code: str) -> Optional[pd.DataFrame]:
        """
        Obtiene datos de estaciones para hacer predicciones (últimas 6 horas + algo más para contexto)
        Args:
            lat: Latitud
            lon: Longitud
            postal_code: Código postal
        Returns:
            DataFrame con datos de las últimas horas
        """
        try:
            # Intentar cargar desde caché primero
            now = datetime.utcnow()
            cache_start = now - timedelta(hours=12)

            cached_data = self.load_station_data(postal_code, cache_start)
            if cached_data is not None and not cached_data.empty:
                # Filtrar solo últimas 6 horas
                recent_data = cached_data[cached_data.index >= now - timedelta(hours=6)]
                if len(recent_data) >= 20:  # Al menos 20 registros de 15 min = 5 horas
                    logger.info("Usando datos de estaciones del caché")
                    return recent_data

            # Si no hay caché válido, descargar datos recientes
            logger.info("Descargando datos recientes de estaciones...")
            df = self.get_recent_station_data(lat, lon, hours=12)

            if df is None:
                return None

            # Interpolar a 15 minutos
            df = self._interpolate_to_15min(df)

            # Guardar en caché
            self.save_station_data(df, postal_code, now - timedelta(hours=12))

            # Retornar solo últimas 6 horas
            recent_6h = df[df.index >= now - timedelta(hours=6)]

            return recent_6h

        except Exception as e:
            logger.error(f"Error obteniendo datos de estaciones para predicción: {e}")
            return None
