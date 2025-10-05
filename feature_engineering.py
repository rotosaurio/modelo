"""
Módulo de ingeniería de características para datos climáticos
Genera variables derivadas como lags, acumulados, tendencias, etc.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class WeatherFeatureEngineer:
    """Clase para generar características derivadas de datos climáticos"""

    def __init__(self):
        self.generated_features = []

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características temporales (hora del día, día de la semana, mes, etc.)
        Args:
            df: DataFrame con índice de timestamp
        Returns:
            DataFrame con características temporales añadidas
        """
        try:
            df_features = df.copy()

            # Crear diccionario con todas las nuevas columnas para evitar fragmentación
            new_columns = {}

            # Extraer componentes temporales
            new_columns['hour'] = df_features.index.hour
            new_columns['day_of_week'] = df_features.index.dayofweek
            new_columns['month'] = df_features.index.month
            new_columns['day_of_year'] = df_features.index.dayofyear

            # Hora en formato cíclico (seno y coseno)
            hours_radians = 2 * np.pi * new_columns['hour'] / 24
            new_columns['hour_sin'] = np.sin(hours_radians)
            new_columns['hour_cos'] = np.cos(hours_radians)

            # Día de la semana en formato cíclico
            dow_radians = 2 * np.pi * new_columns['day_of_week'] / 7
            new_columns['dow_sin'] = np.sin(dow_radians)
            new_columns['dow_cos'] = np.cos(dow_radians)

            # Mes en formato cíclico
            month_radians = 2 * np.pi * (new_columns['month'] - 1) / 12
            new_columns['month_sin'] = np.sin(month_radians)
            new_columns['month_cos'] = np.cos(month_radians)

            # Estación del año
            new_columns['season'] = pd.cut(new_columns['month'],
                                         bins=[0, 3, 6, 9, 12],
                                         labels=['winter', 'spring', 'summer', 'fall'],
                                         include_lowest=True)

            # Codificación one-hot de estación
            season_dummies = pd.get_dummies(new_columns['season'], prefix='season')
            season_df = pd.DataFrame(season_dummies, index=df_features.index)

            # Añadir todas las columnas de una vez usando concat (más eficiente)
            df_features = pd.concat([df_features, pd.DataFrame(new_columns, index=df_features.index), season_df], axis=1)

            logger.info(f"Características temporales creadas: {len(df_features.columns) - len(df.columns)} nuevas variables")
            return df_features

        except Exception as e:
            logger.error(f"Error creando características temporales: {e}")
            return df

    def create_lag_features(self, df: pd.DataFrame, variables: List[str],
                           lags: List[int] = [1, 2, 3, 6]) -> pd.DataFrame:
        """
        Crea características de lag (valores anteriores)
        Args:
            df: DataFrame con datos
            variables: Variables para crear lags
            lags: Lista de lags en pasos de 15 minutos
        Returns:
            DataFrame con características de lag
        """
        try:
            df_features = df.copy()

            # Crear diccionario con todas las nuevas columnas para evitar fragmentación
            new_columns = {}

            for var in variables:
                if var in df_features.columns:
                    for lag in lags:
                        lag_col = f"{var}_lag_{lag}"
                        new_columns[lag_col] = df_features[var].shift(lag)

            # Añadir todas las columnas de una vez usando concat (más eficiente)
            if new_columns:
                df_features = pd.concat([df_features, pd.DataFrame(new_columns, index=df_features.index)], axis=1)

            # Número de lags creados
            lag_features_count = len(new_columns)
            logger.info(f"Características de lag creadas: {lag_features_count} variables")

            return df_features

        except Exception as e:
            logger.error(f"Error creando características de lag: {e}")
            return df

    def create_rolling_features(self, df: pd.DataFrame, variables: List[str],
                               windows: List[int] = [4, 8, 12]) -> pd.DataFrame:
        """
        Crea características de rolling statistics (media, std, min, max)
        Args:
            df: DataFrame con datos
            variables: Variables para crear estadísticas rolling
            windows: Ventanas en pasos de 15 minutos
        Returns:
            DataFrame con características rolling
        """
        try:
            df_features = df.copy()

            # Crear diccionario con todas las nuevas columnas para evitar fragmentación
            new_columns = {}

            for var in variables:
                if var in df_features.columns:
                    for window in windows:
                        # Media móvil
                        new_columns[f"{var}_rolling_mean_{window}"] = df_features[var].rolling(window=window).mean()

                        # Desviación estándar móvil
                        new_columns[f"{var}_rolling_std_{window}"] = df_features[var].rolling(window=window).std()

                        # Mínimo móvil
                        new_columns[f"{var}_rolling_min_{window}"] = df_features[var].rolling(window=window).min()

                        # Máximo móvil
                        new_columns[f"{var}_rolling_max_{window}"] = df_features[var].rolling(window=window).max()

            # Añadir todas las columnas de una vez usando concat (más eficiente)
            if new_columns:
                df_features = pd.concat([df_features, pd.DataFrame(new_columns, index=df_features.index)], axis=1)

            # Número de rolling features creados
            rolling_features_count = len(new_columns)
            logger.info(f"Características rolling creadas: {rolling_features_count} variables")

            return df_features

        except Exception as e:
            logger.error(f"Error creando características rolling: {e}")
            return df

    def create_cumulative_features(self, df: pd.DataFrame, variables: List[str],
                                 periods: List[int] = [4, 8, 12]) -> pd.DataFrame:
        """
        Crea características acumuladas (suma, promedio) en diferentes períodos
        Args:
            df: DataFrame con datos
            variables: Variables para crear acumuladas
            periods: Períodos en pasos de 15 minutos
        Returns:
            DataFrame con características acumuladas
        """
        try:
            df_features = df.copy()

            # Crear diccionario con todas las nuevas columnas para evitar fragmentación
            new_columns = {}

            for var in variables:
                if var in df_features.columns:
                    for period in periods:
                        # Suma acumulada en período
                        new_columns[f"{var}_cumsum_{period}"] = df_features[var].rolling(window=period).sum()

                        # Promedio acumulado en período
                        new_columns[f"{var}_cummean_{period}"] = df_features[var].rolling(window=period).mean()

            # Añadir todas las columnas de una vez usando concat (más eficiente)
            if new_columns:
                df_features = pd.concat([df_features, pd.DataFrame(new_columns, index=df_features.index)], axis=1)

            # Número de cumulative features creados
            cum_features_count = len(new_columns)
            logger.info(f"Características acumuladas creadas: {cum_features_count} variables")

            return df_features

        except Exception as e:
            logger.error(f"Error creando características acumuladas: {e}")
            return df

    def create_difference_features(self, df: pd.DataFrame, variables: List[str],
                                 diffs: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """
        Crea características de diferencias (cambios) respecto a períodos anteriores
        Args:
            df: DataFrame con datos
            variables: Variables para crear diferencias
            diffs: Diferencias en pasos de 15 minutos
        Returns:
            DataFrame con características de diferencias
        """
        try:
            df_features = df.copy()

            # Crear diccionario con todas las nuevas columnas para evitar fragmentación
            new_columns = {}

            for var in variables:
                if var in df_features.columns:
                    for diff in diffs:
                        # Diferencia respecto a diff pasos atrás
                        new_columns[f"{var}_diff_{diff}"] = df_features[var].diff(periods=diff)

                        # Tasa de cambio (diferencia por unidad de tiempo)
                        new_columns[f"{var}_rate_{diff}"] = new_columns[f"{var}_diff_{diff}"] / diff

            # Añadir todas las columnas de una vez usando concat (más eficiente)
            if new_columns:
                df_features = pd.concat([df_features, pd.DataFrame(new_columns, index=df_features.index)], axis=1)

            # Número de difference features creados
            diff_features_count = len(new_columns)
            logger.info(f"Características de diferencias creadas: {diff_features_count} variables")

            return df_features

        except Exception as e:
            logger.error(f"Error creando características de diferencias: {e}")
            return df

    def create_weather_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características específicas del clima
        Args:
            df: DataFrame con variables climáticas
        Returns:
            DataFrame con características específicas del clima
        """
        try:
            df_features = df.copy()

            # Variables base
            temp_col = 'temp_celsius'
            humidity_col = 'relative_humidity_percent'
            precip_col = 'precipitation_mm'
            wind_col = 'wind_speed_ms'
            pressure_col = 'pressure_hpa'

            # Índice de calor (Heat Index) - aproximación
            if temp_col in df_features.columns and humidity_col in df_features.columns:
                T = df_features[temp_col]
                RH = df_features[humidity_col]

                # Fórmula simplificada del Heat Index
                df_features['heat_index'] = (
                    -8.78469475556 + 1.61139411 * T + 2.33854883889 * RH
                    - 0.14611605 * T * RH - 0.012308094 * T**2
                    - 0.01642482778 * RH**2 + 0.002211732 * T**2 * RH
                    + 0.00072546 * T * RH**2 - 0.000003582 * T**2 * RH**2
                )

            # Punto de rocío aproximado
            if temp_col in df_features.columns and humidity_col in df_features.columns:
                T = df_features[temp_col]
                RH = df_features[humidity_col] / 100  # Convertir a fracción

                # Fórmula de Magnus para punto de rocío
                alpha = np.log(RH) + (17.625 * T) / (243.04 + T)
                df_features['dewpoint_approx'] = (243.04 * alpha) / (17.625 - alpha)

            # Comfort térmico (sensación térmica simple)
            if temp_col in df_features.columns and wind_col in df_features.columns:
                # Aproximación simple del wind chill
                T = df_features[temp_col]
                V = df_features[wind_col]

                # Wind chill solo si T < 10°C y V > 1.3 m/s
                mask = (T < 10) & (V > 1.3)
                df_features['wind_chill'] = T.copy()
                df_features.loc[mask, 'wind_chill'] = (
                    13.12 + 0.6215 * T - 11.37 * V**0.16 + 0.3965 * T * V**0.16
                )

            # Cambio de presión (tendencia barométrica)
            if pressure_col in df_features.columns:
                df_features['pressure_change_3h'] = df_features[pressure_col].diff(periods=12)  # 12 * 15min = 3h

            # Intensidad de precipitación por hora
            if precip_col in df_features.columns:
                df_features['precip_intensity'] = df_features[precip_col] * 4  # mm/15min -> mm/h

            # Categorías de intensidad de precipitación
            if 'precip_intensity' in df_features.columns:
                conditions = [
                    (df_features['precip_intensity'] == 0),
                    (df_features['precip_intensity'] < 2.5),
                    (df_features['precip_intensity'] < 7.6),
                    (df_features['precip_intensity'] >= 7.6)
                ]
                choices = ['no_rain', 'light_rain', 'moderate_rain', 'heavy_rain']
                df_features['precip_category'] = np.select(conditions, choices, default='unknown')

                # One-hot encoding
                precip_dummies = pd.get_dummies(df_features['precip_category'], prefix='precip')
                df_features = pd.concat([df_features, precip_dummies], axis=1)
                df_features = df_features.drop('precip_category', axis=1)

            # Categorías de nubosidad
            cloud_col = 'cloud_cover_percent'
            if cloud_col in df_features.columns:
                conditions = [
                    (df_features[cloud_col] < 25),
                    (df_features[cloud_col] < 50),
                    (df_features[cloud_col] < 75),
                    (df_features[cloud_col] >= 75)
                ]
                choices = ['clear', 'partly_cloudy', 'mostly_cloudy', 'overcast']
                df_features['cloud_category'] = np.select(conditions, choices, default='unknown')

                # One-hot encoding
                cloud_dummies = pd.get_dummies(df_features['cloud_category'], prefix='cloud')
                df_features = pd.concat([df_features, cloud_dummies], axis=1)
                df_features = df_features.drop('cloud_category', axis=1)

            weather_features_count = len(df_features.columns) - len(df.columns)
            logger.info(f"Características específicas del clima creadas: {weather_features_count} variables")

            return df_features

        except Exception as e:
            logger.error(f"Error creando características específicas del clima: {e}")
            return df

    def create_all_features(self, df: pd.DataFrame, variables: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Crea todas las características derivadas
        Args:
            df: DataFrame original
            variables: Variables base para crear features (si None, usa todas numéricas)
        Returns:
            DataFrame con todas las características
        """
        try:
            if variables is None:
                variables = df.select_dtypes(include=[np.number]).columns.tolist()

            logger.info(f"Creando características derivadas para {len(variables)} variables base")

            # 1. Características temporales
            df_features = self.create_temporal_features(df)

            # 2. Características de lag
            df_features = self.create_lag_features(df_features, variables)

            # 3. Características rolling
            df_features = self.create_rolling_features(df_features, variables)

            # 4. Características acumuladas
            df_features = self.create_cumulative_features(df_features, variables)

            # 5. Características de diferencias
            df_features = self.create_difference_features(df_features, variables)

            # 6. Características específicas del clima
            df_features = self.create_weather_specific_features(df_features)

            # Registrar features generadas
            original_cols = set(df.columns)
            self.generated_features = [col for col in df_features.columns if col not in original_cols]

            logger.info(f"Total de características creadas: {len(self.generated_features)}")
            logger.info(f"Dataset final: {df_features.shape[0]} filas x {df_features.shape[1]} columnas")

            return df_features

        except Exception as e:
            logger.error(f"Error creando todas las características: {e}")
            return df

    def select_features_for_model(self, df: pd.DataFrame, max_features: int = 20) -> pd.DataFrame:
        """
        Selecciona las características más importantes para el modelo (versión simplificada)
        Args:
            df: DataFrame con todas las características
            max_features: Número máximo de características a mantener
        Returns:
            DataFrame con características seleccionadas
        """
        try:
            # Variables base más importantes
            base_vars = ['temp_celsius', 'relative_humidity_percent', 'precipitation_mm',
                        'wind_speed_ms', 'pressure_hpa', 'cloud_cover_percent']

            # Características temporales básicas
            temporal_vars = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']

            # Lags muy básicos (solo para variables más importantes)
            important_lags = []
            for var in base_vars[:3]:  # Solo temp, humidity, precip
                for lag in [1, 2]:     # Solo lags muy cortos
                    lag_col = f"{var}_lag_{lag}"
                    if lag_col in df.columns:
                        important_lags.append(lag_col)

            # Características específicas del clima importantes
            weather_specific = ['heat_index', 'pressure_change_3h']

            # Crear lista de características priorizadas en orden de importancia
            priority_features = base_vars + temporal_vars + important_lags + weather_specific

            # Filtrar las que existen en el DataFrame
            existing_priority = [col for col in priority_features if col in df.columns]

            # Si necesitamos más características, añadir algunas adicionales selectivamente
            if len(existing_priority) < max_features:
                # Añadir algunas características de diferencia básicas si están disponibles
                diff_features = [col for col in df.columns if '_diff_1' in col and col.split('_diff_')[0] in base_vars[:3]]
                existing_priority.extend(diff_features[:max_features - len(existing_priority)])

            # Si aún necesitamos más, añadir algunas características acumuladas básicas
            if len(existing_priority) < max_features:
                cumsum_features = [col for col in df.columns if '_cumsum_4' in col and col.split('_cumsum_')[0] in base_vars[:3]]
                existing_priority.extend(cumsum_features[:max_features - len(existing_priority)])

            # Limitar al número máximo de características
            if len(existing_priority) > max_features:
                existing_priority = existing_priority[:max_features]

            # Seleccionar columnas
            df_selected = df[existing_priority].copy()

            logger.info(f"Características seleccionadas: {len(existing_priority)} de {len(df.columns)}")
            return df_selected

        except Exception as e:
            logger.error(f"Error seleccionando características: {e}")
            return df
