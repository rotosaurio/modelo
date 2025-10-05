"""
Módulo para descargar datos de precipitación IMERG de NASA
Utiliza NASA Giovanni y Earthdata para acceso a datos GPM IMERG
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import requests
import xarray as xr
import numpy as np
import pandas as pd
from pydap.client import open_url
from pydap.cas.urs import setup_session

from config import WEATHER_DATA_DIR, IMERG_CONFIG, NASA_CONFIG
from utils_credentials import credentials_manager

logger = logging.getLogger(__name__)


class IMERGDataDownloader:
    """Clase para descargar datos IMERG de precipitación"""

    def __init__(self):
        self.data_dir = WEATHER_DATA_DIR / "imerg"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session = None

    def _setup_nasa_session(self) -> bool:
        """
        Configura sesión autenticada con NASA Earthdata usando bearer token
        Returns:
            True si se configuró correctamente
        """
        try:
            # Usar el gestor de credenciales para obtener sesión con bearer token
            self.session = credentials_manager.setup_nasa_session()

            if self.session:
                logger.info("Sesión NASA configurada correctamente con bearer token")
                return True
            else:
                logger.warning("No se pudo configurar sesión NASA")
                logger.info("Verifica que NASA_EARTHDATA_BEARER_TOKEN esté configurado en .env")
                return False

        except Exception as e:
            logger.error(f"Error configurando sesión NASA: {e}")
            return False

    def _get_imerg_url(self, date: datetime) -> str:
        """
        Construye URL para datos IMERG de una fecha específica
        Args:
            date: Fecha y hora deseada
        Returns:
            URL completa para descarga
        """
        # Formato: YYYY/MM/GPM_3IMERGHH.06_YYYYMMDD-HHMMSS.nc4
        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")
        hour = date.strftime("%H")
        minute = date.strftime("%M")
        second = date.strftime("%S")

        filename = f"GPM_3IMERGHH.06_{year}{month}{day}-S{hour}{minute}{second}-E{hour}{minute}{second}.RT-H5"
        url = f"{IMERG_CONFIG['base_url']}/{year}/{month}/{filename}"

        return url

    def download_imerg_data(self, lat: float, lon: float, start_date: datetime,
                           end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Descarga datos IMERG para coordenadas específicas en un rango de fechas
        Args:
            lat: Latitud
            lon: Longitud
            start_date: Fecha de inicio
            end_date: Fecha de fin
        Returns:
            DataFrame con datos de precipitación o None si falla
        """
        try:
            if not self._setup_nasa_session():
                logger.error("No se pudo configurar sesión con NASA")
                return None

            logger.info(f"Descargando datos IMERG para ({lat:.3f}, {lon:.3f}) del {start_date} al {end_date}")

            all_data = []

            # IMERG tiene resolución de 30 minutos
            current_date = start_date
            while current_date <= end_date:
                try:
                    url = self._get_imerg_url(current_date)

                    # Intentar abrir dataset con pydap
                    dataset = open_url(url, session=self.session)

                    # Extraer datos de precipitación
                    precip_data = dataset['precipitationCal']

                    # Crear coordenadas para el punto específico
                    lat_idx = np.argmin(np.abs(precip_data.lat[:] - lat))
                    lon_idx = np.argmin(np.abs(precip_data.lon[:] - lon))

                    # Extraer valor para el punto
                    precip_value = float(precip_data[lat_idx, lon_idx])

                    # Crear registro
                    record = {
                        'timestamp': current_date,
                        'latitude': lat,
                        'longitude': lon,
                        'precipitation_mm': max(0, precip_value)  # Asegurar no negativo
                    }

                    all_data.append(record)
                    logger.debug(f"Descargado: {current_date} - {precip_value:.3f} mm")

                except Exception as e:
                    logger.warning(f"Error descargando datos para {current_date}: {e}")
                    # Continuar con siguiente timestamp

                # Avanzar 30 minutos
                current_date += timedelta(minutes=30)

            if not all_data:
                logger.error("No se pudieron descargar datos IMERG")
                return None

            # Crear DataFrame
            df = pd.DataFrame(all_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()

            logger.info(f"Descargados {len(df)} registros IMERG")
            return df

        except Exception as e:
            logger.error(f"Error descargando datos IMERG: {e}")
            return None

    def get_recent_imerg_data(self, lat: float, lon: float, hours: int = 24) -> Optional[pd.DataFrame]:
        """
        Descarga datos IMERG recientes para predicción
        Args:
            lat: Latitud
            lon: Longitud
            hours: Número de horas hacia atrás
        Returns:
            DataFrame con datos recientes
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=hours)

        return self.download_imerg_data(lat, lon, start_date, end_date)

    def _interpolate_to_15min(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpola datos de 30 minutos a 15 minutos
        Args:
            df: DataFrame con datos de 30 minutos
        Returns:
            DataFrame con datos de 15 minutos
        """
        try:
            # Reindexar a 15 minutos
            df_15min = df.resample('15min').interpolate(method='linear')

            # Para precipitación, usar interpolación más conservadora
            # La precipitación se acumula, no se interpola linealmente
            df_15min['precipitation_mm'] = df_15min['precipitation_mm'].clip(lower=0)

            logger.info(f"Datos interpolados de {len(df)} a {len(df_15min)} registros (15min)")
            return df_15min

        except Exception as e:
            logger.error(f"Error interpolando datos: {e}")
            return df

    def save_imerg_data(self, df: pd.DataFrame, postal_code: str, start_date: datetime) -> Path:
        """
        Guarda datos IMERG en archivo
        Args:
            df: DataFrame con datos
            postal_code: Código postal
            start_date: Fecha de inicio
        Returns:
            Path al archivo guardado
        """
        filename = f"imerg_{postal_code}_{start_date.strftime('%Y%m%d_%H%M')}.csv"
        filepath = self.data_dir / filename

        df.to_csv(filepath)
        logger.info(f"Datos IMERG guardados: {filepath}")

        return filepath

    def load_imerg_data(self, postal_code: str, start_date: datetime) -> Optional[pd.DataFrame]:
        """
        Carga datos IMERG desde archivo
        Args:
            postal_code: Código postal
            start_date: Fecha de inicio
        Returns:
            DataFrame con datos o None si no existe
        """
        pattern = f"imerg_{postal_code}_{start_date.strftime('%Y%m%d_%H%M')}*.csv"
        files = list(self.data_dir.glob(pattern))

        if not files:
            return None

        try:
            df = pd.read_csv(files[0], index_col='timestamp', parse_dates=True)
            logger.info(f"Datos IMERG cargados: {files[0]}")
            return df
        except Exception as e:
            logger.error(f"Error cargando datos IMERG: {e}")
            return None

    def get_imerg_for_prediction(self, lat: float, lon: float, postal_code: str) -> Optional[pd.DataFrame]:
        """
        Obtiene datos IMERG para hacer predicciones (últimas 6 horas + algo más para contexto)
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
            cache_start = now - timedelta(hours=12)  # 12 horas para tener buffer

            cached_data = self.load_imerg_data(postal_code, cache_start)
            if cached_data is not None and not cached_data.empty:
                # Filtrar solo últimas 6 horas
                recent_data = cached_data[cached_data.index >= now - timedelta(hours=6)]
                if len(recent_data) >= 20:  # Al menos 20 registros de 15 min = 5 horas
                    logger.info("Usando datos IMERG del caché")
                    return recent_data

            # Si no hay caché válido, descargar datos recientes
            logger.info("Descargando datos IMERG recientes...")
            df = self.get_recent_imerg_data(lat, lon, hours=12)

            if df is None:
                return None

            # Interpolar a 15 minutos
            df = self._interpolate_to_15min(df)

            # Guardar en caché
            self.save_imerg_data(df, postal_code, now - timedelta(hours=12))

            # Retornar solo últimas 6 horas
            recent_6h = df[df.index >= now - timedelta(hours=6)]

            return recent_6h

        except Exception as e:
            logger.error(f"Error obteniendo datos IMERG para predicción: {e}")
            return None
