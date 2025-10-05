"""
Módulo para descargar datos ERA5/ERA5-Land de Copernicus CDS
Incluye temperatura, humedad, viento, presión y cobertura de nubes
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import cdsapi
import xarray as xr
import numpy as np
import pandas as pd

from config import WEATHER_DATA_DIR, ERA5_CONFIG, CDS_API_CONFIG
from utils_credentials import credentials_manager

logger = logging.getLogger(__name__)


class ERA5DataDownloader:
    """Clase para descargar datos ERA5 de Copernicus"""

    def __init__(self):
        self.data_dir = WEATHER_DATA_DIR / "era5"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.client = None

    def _setup_cds_client(self) -> bool:
        """
        Configura cliente de Copernicus CDS API
        Soporta tanto .env como archivos legacy
        Returns:
            True si se configuró correctamente
        """
        try:
            # Intentar obtener credenciales del gestor unificado
            api_key = credentials_manager.get_cds_credentials()

            if api_key:
                # Configurar cliente con credenciales directas
                self.client = cdsapi.Client(key=api_key, url=CDS_API_CONFIG["url"])
                logger.info("Cliente CDS configurado correctamente con credenciales de .env")
                return True
            else:
                logger.warning("No se encontraron credenciales para CDS API")
                logger.info("Para configurar credenciales:")
                logger.info("Opción 1 - Archivo .env:")
                logger.info("   Crea archivo .env con: CDS_API_KEY=tu_uid:tu_api_key")
                logger.info("Opción 2 - Archivo legacy ~/.cdsapirc:")
                logger.info("   Crea ~/.cdsapirc con:")
                logger.info("   url: https://cds.climate.copernicus.eu/api/v2")
                logger.info("   key: TU_UID:TU_API_KEY")
                logger.info("Opción 3 - Variables de entorno:")
                logger.info("   export CDS_API_KEY=tu_uid:tu_api_key")
                return False

        except Exception as e:
            logger.error(f"Error configurando cliente CDS: {e}")
            return False

    def _build_request(self, lat: float, lon: float, start_date: datetime,
                      end_date: datetime) -> Dict[str, Any]:
        """
        Construye solicitud para CDS API
        Args:
            lat: Latitud
            lon: Longitud
            start_date: Fecha de inicio
            end_date: Fecha de fin
        Returns:
            Diccionario con parámetros de la solicitud
        """
        # Definir área (bounding box alrededor del punto)
        # ERA5 tiene resolución de ~0.25 grados, así que tomamos un área pequeña
        area = [
            lat + 0.5,  # north
            lon - 0.5,  # west
            lat - 0.5,  # south
            lon + 0.5   # east
        ]

        request = {
            'product_type': ERA5_CONFIG['product_type'],
            'format': ERA5_CONFIG['format'],
            'variable': ERA5_CONFIG['variables'],
            'year': start_date.strftime('%Y'),
            'month': start_date.strftime('%m'),
            'day': [f"{d:02d}" for d in range(start_date.day, end_date.day + 1)],
            'time': [f"{h:02d}:00" for h in range(24)],  # Todas las horas
            'area': area
        }

        return request

    def download_era5_data(self, lat: float, lon: float, start_date: datetime,
                          end_date: datetime) -> Optional[xr.Dataset]:
        """
        Descarga datos ERA5 para coordenadas específicas
        Args:
            lat: Latitud
            lon: Longitud
            start_date: Fecha de inicio
            end_date: Fecha de fin
        Returns:
            Dataset de xarray con datos ERA5 o None si falla
        """
        try:
            if not self._setup_cds_client():
                logger.error("No se pudo configurar cliente CDS")
                return None

            logger.info(f"Descargando datos ERA5 para ({lat:.3f}, {lon:.3f}) del {start_date.date()} al {end_date.date()}")

            request = self._build_request(lat, lon, start_date, end_date)

            # Generar nombre de archivo temporal
            filename = f"era5_{lat:.3f}_{lon:.3f}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.nc"
            filepath = self.data_dir / filename

            # Descargar datos
            self.client.retrieve(
                ERA5_CONFIG['dataset'],
                request,
                str(filepath)
            )

            # Cargar datos
            ds = xr.open_dataset(filepath)

            logger.info(f"Datos ERA5 descargados: {filepath}")
            return ds

        except Exception as e:
            logger.error(f"Error descargando datos ERA5: {e}")
            return None

    def process_era5_data(self, ds: xr.Dataset, target_lat: float,
                         target_lon: float) -> pd.DataFrame:
        """
        Procesa dataset ERA5 y extrae datos para coordenadas específicas
        Args:
            ds: Dataset de ERA5
            target_lat: Latitud objetivo
            target_lon: Longitud objetivo
        Returns:
            DataFrame con datos procesados
        """
        try:
            logger.info("Procesando datos ERA5...")

            # Encontrar índices más cercanos a las coordenadas objetivo
            lat_idx = np.argmin(np.abs(ds.latitude.values - target_lat))
            lon_idx = np.argmin(np.abs(ds.longitude.values - target_lon))

            # Extraer datos para el punto específico
            data = ds.isel(latitude=lat_idx, longitude=lon_idx)

            # Convertir a DataFrame
            df = data.to_dataframe()

            # Resetear índice para tener timestamp como columna
            df = df.reset_index()

            # Convertir temperaturas de Kelvin a Celsius
            if 't2m' in df.columns:
                df['temperature_celsius'] = df['t2m'] - 273.15
            elif '2m_temperature' in df.columns:
                df['temperature_celsius'] = df['2m_temperature'] - 273.15

            # Calcular velocidad del viento
            if 'u10' in df.columns and 'v10' in df.columns:
                df['wind_speed_ms'] = np.sqrt(df['u10']**2 + df['v10']**2)
                df['wind_direction_deg'] = np.degrees(np.arctan2(df['v10'], df['u10'])) % 360
            elif '10m_u_component_of_wind' in df.columns and '10m_v_component_of_wind' in df.columns:
                df['wind_speed_ms'] = np.sqrt(df['10m_u_component_of_wind']**2 + df['10m_v_component_of_wind']**2)
                df['wind_direction_deg'] = np.degrees(np.arctan2(df['10m_v_component_of_wind'], df['10m_u_component_of_wind'])) % 360

            # Renombrar columnas para consistencia
            column_mapping = {
                'sp': 'pressure_hpa',
                'surface_pressure': 'pressure_hpa',
                'rh': 'relative_humidity_percent',
                'relative_humidity': 'relative_humidity_percent',
                'tcc': 'cloud_cover_percent',
                'total_cloud_cover': 'cloud_cover_percent',
                'd2m': 'dewpoint_celsius',
                '2m_dewpoint_temperature': 'dewpoint_celsius'
            }

            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df[new_col] = df[old_col]

            # Convertir unidades si es necesario
            if 'pressure_hpa' in df.columns:
                df['pressure_hpa'] = df['pressure_hpa'] / 100  # Pa to hPa

            # Calcular humedad relativa si no está disponible
            if 'relative_humidity_percent' not in df.columns and 'dewpoint_celsius' in df.columns and 'temperature_celsius' in df.columns:
                # Fórmula de Magnus para humedad relativa aproximada
                df['relative_humidity_percent'] = 100 * np.exp(
                    (17.625 * df['dewpoint_celsius']) / (243.04 + df['dewpoint_celsius'])
                ) / np.exp(
                    (17.625 * df['temperature_celsius']) / (243.04 + df['temperature_celsius'])
                )

            # Seleccionar columnas finales
            final_columns = [
                'valid_time', 'temperature_celsius', 'relative_humidity_percent',
                'wind_speed_ms', 'wind_direction_deg', 'pressure_hpa', 'cloud_cover_percent'
            ]

            available_columns = [col for col in final_columns if col in df.columns]
            df_final = df[available_columns].copy()

            # Renombrar valid_time a timestamp
            if 'valid_time' in df_final.columns:
                df_final = df_final.rename(columns={'valid_time': 'timestamp'})

            # Establecer timestamp como índice
            if 'timestamp' in df_final.columns:
                df_final['timestamp'] = pd.to_datetime(df_final['timestamp'])
                df_final = df_final.set_index('timestamp').sort_index()

            logger.info(f"Datos ERA5 procesados: {len(df_final)} registros")
            return df_final

        except Exception as e:
            logger.error(f"Error procesando datos ERA5: {e}")
            return pd.DataFrame()

    def get_recent_era5_data(self, lat: float, lon: float, hours: int = 24) -> Optional[pd.DataFrame]:
        """
        Descarga datos ERA5 recientes para predicción
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

            # Descargar datos
            ds = self.download_era5_data(lat, lon, start_date, end_date)
            if ds is None:
                return None

            # Procesar datos
            df = self.process_era5_data(ds, lat, lon)

            return df

        except Exception as e:
            logger.error(f"Error obteniendo datos ERA5 recientes: {e}")
            return None

    def _interpolate_to_15min(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpola datos de 1 hora a 15 minutos
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

            if 'wind_speed_ms' in df_15min.columns:
                df_15min['wind_speed_ms'] = df_15min['wind_speed_ms'].clip(0, 100)

            if 'pressure_hpa' in df_15min.columns:
                df_15min['pressure_hpa'] = df_15min['pressure_hpa'].clip(800, 1100)

            if 'cloud_cover_percent' in df_15min.columns:
                df_15min['cloud_cover_percent'] = df_15min['cloud_cover_percent'].clip(0, 100)

            logger.info(f"Datos interpolados de {len(df)} a {len(df_15min)} registros (15min)")
            return df_15min

        except Exception as e:
            logger.error(f"Error interpolando datos ERA5: {e}")
            return df

    def save_era5_data(self, df: pd.DataFrame, postal_code: str, start_date: datetime) -> Path:
        """
        Guarda datos ERA5 en archivo
        Args:
            df: DataFrame con datos
            postal_code: Código postal
            start_date: Fecha de inicio
        Returns:
            Path al archivo guardado
        """
        filename = f"era5_{postal_code}_{start_date.strftime('%Y%m%d_%H%M')}.csv"
        filepath = self.data_dir / filename

        df.to_csv(filepath)
        logger.info(f"Datos ERA5 guardados: {filepath}")

        return filepath

    def load_era5_data(self, postal_code: str, start_date: datetime) -> Optional[pd.DataFrame]:
        """
        Carga datos ERA5 desde archivo
        Args:
            postal_code: Código postal
            start_date: Fecha de inicio
        Returns:
            DataFrame con datos o None si no existe
        """
        pattern = f"era5_{postal_code}_{start_date.strftime('%Y%m%d_%H%M')}*.csv"
        files = list(self.data_dir.glob(pattern))

        if not files:
            return None

        try:
            df = pd.read_csv(files[0], index_col='timestamp', parse_dates=True)
            logger.info(f"Datos ERA5 cargados: {files[0]}")
            return df
        except Exception as e:
            logger.error(f"Error cargando datos ERA5: {e}")
            return None

    def get_era5_for_prediction(self, lat: float, lon: float, postal_code: str) -> Optional[pd.DataFrame]:
        """
        Obtiene datos ERA5 para hacer predicciones (últimas 6 horas + algo más para contexto)
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

            cached_data = self.load_era5_data(postal_code, cache_start)
            if cached_data is not None and not cached_data.empty:
                # Filtrar solo últimas 6 horas
                recent_data = cached_data[cached_data.index >= now - timedelta(hours=6)]
                if len(recent_data) >= 20:  # Al menos 20 registros de 15 min = 5 horas
                    logger.info("Usando datos ERA5 del caché")
                    return recent_data

            # Si no hay caché válido, descargar datos recientes
            logger.info("Descargando datos ERA5 recientes...")
            df = self.get_recent_era5_data(lat, lon, hours=12)

            if df is None:
                return None

            # Interpolar a 15 minutos
            df = self._interpolate_to_15min(df)

            # Guardar en caché
            self.save_era5_data(df, postal_code, now - timedelta(hours=12))

            # Retornar solo últimas 6 horas
            recent_6h = df[df.index >= now - timedelta(hours=6)]

            return recent_6h

        except Exception as e:
            logger.error(f"Error obteniendo datos ERA5 para predicción: {e}")
            return None
