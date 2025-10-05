"""
Módulo para generar datos climáticos sintéticos
Útil cuando las APIs reales no están disponibles para desarrollo/testing
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple

from config import WEATHER_VARIABLES

logger = logging.getLogger(__name__)


class SyntheticWeatherGenerator:
    """
    Generador de datos climáticos sintéticos basados en patrones reales de Chihuahua
    """

    def __init__(self):
        # Parámetros climáticos típicos de Chihuahua por región
        self.chihuahua_patterns = {
            # Chihuahua capital - continental templado
            "chihuahua_city": {
                "temp_base": 18.0,  # °C promedio anual
                "temp_amplitude": 12.0,  # variación estacional
                "humidity_base": 45.0,  # % promedio
                "precipitation_monthly": [5, 8, 3, 2, 5, 40, 80, 70, 50, 25, 10, 5],  # mm por mes
                "wind_speed_base": 8.0,  # km/h
                "pressure_base": 1013.0,  # hPa
            },
            # Ciudad Juárez - desierto
            "juarez": {
                "temp_base": 20.0,
                "temp_amplitude": 15.0,
                "humidity_base": 35.0,
                "precipitation_monthly": [3, 5, 2, 2, 5, 8, 35, 30, 20, 10, 5, 3],
                "wind_speed_base": 12.0,
                "pressure_base": 1010.0,
            },
            # Zona serrana (Cuauhtémoc, Creel)
            "mountains": {
                "temp_base": 12.0,
                "temp_amplitude": 8.0,
                "humidity_base": 55.0,
                "precipitation_monthly": [10, 15, 8, 5, 15, 60, 120, 100, 80, 40, 15, 10],
                "wind_speed_base": 15.0,
                "pressure_base": 980.0,
            }
        }

    def get_region_for_coords(self, lat: float, lon: float) -> str:
        """
        Determina la región climática basada en coordenadas
        """
        # Ciudad Juárez (norte, más caluroso)
        if lat > 31.0:
            return "juarez"
        # Zona serrana (oeste, más frío y húmedo)
        elif lon < -106.5:
            return "mountains"
        # Chihuahua capital y zona central
        else:
            return "chihuahua_city"

    def generate_synthetic_weather(self, lat: float, lon: float,
                                 start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Genera datos climáticos sintéticos para coordenadas específicas
        Args:
            lat: Latitud
            lon: Longitud
            start_date: Fecha de inicio
            end_date: Fecha de fin
        Returns:
            DataFrame con datos sintéticos
        """
        try:
            logger.info(f"Generando datos sintéticos para ({lat:.3f}, {lon:.3f}) del {start_date.date()} al {end_date.date()}")

            # Determinar región climática
            region = self.get_region_for_coords(lat, lon)
            params = self.chihuahua_patterns[region]

            # Crear índice de tiempo cada 15 minutos
            time_index = pd.date_range(start=start_date, end=end_date, freq='15min')

            # Generar datos base
            data = []

            for ts in time_index:
                # Componentes temporales
                day_of_year = ts.dayofyear
                hour = ts.hour + ts.minute / 60.0  # hora decimal

                # Temperatura con variación diaria y estacional
                seasonal_temp = params["temp_base"] + params["temp_amplitude"] * np.sin(2 * np.pi * (day_of_year - 80) / 365)
                daily_temp = 3 * np.sin(2 * np.pi * (hour - 6) / 24)  # pico a las 18:00
                noise_temp = np.random.normal(0, 2)  # ruido

                temperature = seasonal_temp + daily_temp + noise_temp

                # Humedad relativa (inversa a temperatura, más baja en verano)
                base_humidity = params["humidity_base"]
                seasonal_humidity = -10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
                daily_humidity = -5 * np.sin(2 * np.pi * hour / 24)
                noise_humidity = np.random.normal(0, 3)

                humidity = np.clip(base_humidity + seasonal_humidity + daily_humidity + noise_humidity, 10, 90)

                # Precipitación (basada en patrones mensuales)
                month_idx = ts.month - 1
                monthly_precip = params["precipitation_monthly"][month_idx]

                # Probabilidad de lluvia basada en el mes
                rain_prob = monthly_precip / 100  # normalizar a 0-1
                rain_prob_hourly = rain_prob * (1 + 0.5 * np.sin(2 * np.pi * hour / 24))  # más lluvia de noche

                # Generar lluvia cada 15 minutos
                precipitation = 0
                if np.random.random() < rain_prob_hourly / 96:  # 96 intervalos de 15 min por día
                    # Intensidad exponencial (lluvias ligeras más probables)
                    precipitation = np.random.exponential(2)  # mm/15min, promedio 2mm

                # Viento
                wind_speed = params["wind_speed_base"] + np.random.normal(0, 3)
                wind_speed = max(0, wind_speed)  # no negativo

                # Dirección del viento (predominante de oeste/suroeste en Chihuahua)
                wind_direction = 225 + np.random.normal(0, 45)  # grados
                wind_direction = wind_direction % 360

                # Presión atmosférica
                pressure = params["pressure_base"] + np.random.normal(0, 5)
                pressure = np.clip(pressure, 980, 1030)

                # Nubosidad (correlacionada con humedad y lluvia)
                cloud_base = humidity / 2  # más nubes con más humedad
                if precipitation > 0:
                    cloud_base += 30  # más nubes durante lluvia

                cloud_cover = np.clip(cloud_base + np.random.normal(0, 10), 0, 100)

                # Crear registro
                record = {
                    'timestamp': ts,
                    'latitude': lat,
                    'longitude': lon,
                    WEATHER_VARIABLES['temperature']: temperature,
                    WEATHER_VARIABLES['humidity']: humidity,
                    WEATHER_VARIABLES['precipitation']: precipitation,
                    WEATHER_VARIABLES['wind_speed']: wind_speed * 1000 / 3600,  # convertir km/h a m/s
                    WEATHER_VARIABLES['wind_direction']: wind_direction,
                    WEATHER_VARIABLES['pressure']: pressure,
                    WEATHER_VARIABLES['cloud_cover']: cloud_cover
                }

                data.append(record)

            # Crear DataFrame
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            logger.info(f"Generados {len(df)} registros sintéticos para región '{region}'")
            return df

        except Exception as e:
            logger.error(f"Error generando datos sintéticos: {e}")
            return pd.DataFrame()

    def generate_training_dataset(self, postal_codes: list, start_date: datetime,
                                end_date: datetime) -> pd.DataFrame:
        """
        Genera un dataset completo de entrenamiento con múltiples códigos postales
        Args:
            postal_codes: Lista de códigos postales
            start_date: Fecha de inicio
            end_date: Fecha de fin
        Returns:
            DataFrame combinado con datos de todos los códigos postales
        """
        try:
            logger.info(f"Generando dataset sintético para {len(postal_codes)} códigos postales")

            all_data = []

            for postal_code in postal_codes:
                # Obtener coordenadas (usando el sistema existente)
                from postal_coordinates import PostalCoordinatesManager
                coord_manager = PostalCoordinatesManager()
                coords = coord_manager.get_coordinates(postal_code)

                if not coords:
                    logger.warning(f"No se encontraron coordenadas para {postal_code}, saltando...")
                    continue

                lat, lon = coords

                # Generar datos sintéticos
                df_postal = self.generate_synthetic_weather(lat, lon, start_date, end_date)
                if not df_postal.empty:
                    df_postal['postal_code'] = postal_code
                    all_data.append(df_postal)

            if not all_data:
                logger.error("No se pudieron generar datos sintéticos")
                return pd.DataFrame()

            # Combinar todos los datos
            combined_df = pd.concat(all_data, ignore_index=False)
            logger.info(f"Dataset sintético generado: {len(combined_df)} registros totales")

            return combined_df

        except Exception as e:
            logger.error(f"Error generando dataset de entrenamiento: {e}")
            return pd.DataFrame()


# Función de conveniencia para uso directo
def generate_synthetic_data(lat: float, lon: float, start_date: datetime,
                          end_date: datetime) -> pd.DataFrame:
    """
    Función de conveniencia para generar datos sintéticos
    """
    generator = SyntheticWeatherGenerator()
    return generator.generate_synthetic_weather(lat, lon, start_date, end_date)
