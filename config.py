"""
Configuraciones y constantes para el modelo de predicción meteorológica de Chihuahua
"""

import os
from pathlib import Path

# === CONFIGURACIONES DE DIRECTORIOS ===
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
POSTAL_DATA_DIR = DATA_DIR / "postal_codes"
WEATHER_DATA_DIR = DATA_DIR / "weather"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Crear directorios si no existen
for dir_path in [DATA_DIR, POSTAL_DATA_DIR, WEATHER_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# === CONFIGURACIONES DE CÓDIGOS POSTALES ===
# Estado de Chihuahua (código INEGI: 08)
CHIHUAHUA_STATE_CODE = "08"

# URLs para datos de códigos postales (INEGI)
POSTAL_DATA_URLS = {
    "shapefile": "https://www.inegi.org.mx/contenidos/productos/prod_serv/contenidos/espanol/bvinegi/productos/geografia/carta/CP_2023_v1.zip",
    "csv": "https://www.inegi.org.mx/contenidos/productos/prod_serv/contenidos/espanol/bvinegi/productos/geografia/carta/CP_2023_v1_csv.zip"
}

# === CONFIGURACIONES METEOROLÓGICAS ===
# Ventanas temporales
INPUT_WINDOW_HOURS = 6  # 6 horas de entrada
OUTPUT_WINDOW_HOURS = 6  # 6 horas de predicción
TIME_STEP_MINUTES = 15   # Intervalo de 15 minutos

# Número de pasos
INPUT_STEPS = (INPUT_WINDOW_HOURS * 60) // TIME_STEP_MINUTES  # 24 pasos
OUTPUT_STEPS = (OUTPUT_WINDOW_HOURS * 60) // TIME_STEP_MINUTES  # 24 pasos

# === CONFIGURACIONES DE DATOS CLIMÁTICOS ===

# NASA GPM IMERG
IMERG_CONFIG = {
    "base_url": "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGHH.06",
    "temporal_resolution": "30min",
    "spatial_resolution": "0.1deg",
    "variables": ["precipitationCal"]
}

# ERA5 / ERA5-Land
ERA5_CONFIG = {
    "dataset": "reanalysis-era5-single-levels",
    "product_type": "reanalysis",
    "format": "netcdf",
    "variables": [
        "2m_temperature",           # Temperatura a 2m
        "2m_dewpoint_temperature",  # Temperatura de rocío a 2m
        "surface_pressure",         # Presión superficial
        "total_cloud_cover",        # Cobertura total de nubes
        "10m_u_component_of_wind",  # Componente U del viento a 10m
        "10m_v_component_of_wind",  # Componente V del viento a 10m
        "relative_humidity",        # Humedad relativa
    ],
    "temporal_resolution": "1hour"
}

# === CONFIGURACIONES DE MODELO ===

# Arquitectura del modelo
MODEL_CONFIG = {
    "input_size": None,  # Se calculará dinámicamente basado en features
    "hidden_size": 32,   # Tamaño original que funcionó bien
    "num_layers": 1,     # Una sola capa para simplicidad
    "output_steps": OUTPUT_STEPS,
    "dropout": 0.1,      # Dropout más bajo
    "bidirectional": False  # Deshabilitado para simplicidad inicial
}

# Entrenamiento
TRAINING_CONFIG = {
    "batch_size": 8,   # Batch size más razonable pero conservador
    "learning_rate": 1e-3,
    "num_epochs": 50,  # Más épocas para mejor convergencia
    "patience": 10,    # Más paciencia para encontrar mejor modelo
    "validation_split": 0.15,  # Balance entre train/val
    "test_split": 0.15,        # Más datos para entrenamiento
    "random_seed": 42
}

# === CONFIGURACIONES DE PREPROCESAMIENTO ===

# Variables climáticas
WEATHER_VARIABLES = {
    "temperature": "temp_celsius",
    "humidity": "relative_humidity_percent",
    "wind_speed": "wind_speed_ms",
    "wind_direction": "wind_direction_deg",
    "pressure": "pressure_hpa",
    "precipitation": "precipitation_mm",
    "cloud_cover": "cloud_cover_percent"
}

# Umbrales para clasificación de clima
WEATHER_THRESHOLDS = {
    "rain_light": 0.2,      # mm/h
    "rain_moderate": 2.5,   # mm/h
    "rain_heavy": 7.6,      # mm/h
    "cloudy": 60,           # %
    "cold": 10,             # grados C
    "hot": 30,              # grados C
    "windy": 10             # m/s
}

# === CONFIGURACIONES DE APIs ===

# Copernicus CDS API (para ERA5)
CDS_API_CONFIG = {
    "url": "https://cds.climate.copernicus.eu/api",
    "env_key": "CDS_API_KEY",  # Variable de entorno para la API key
    "use_synthetic_fallback": True  # Usar datos sintéticos si APIs fallan
}

# NASA Earthdata (para IMERG)
NASA_CONFIG = {
    "earthdata_url": "https://urs.earthdata.nasa.gov",
    "token_url": "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGHH.06",
    "env_bearer_token": "NASA_EARTHDATA_BEARER_TOKEN",
    "use_synthetic_fallback": True  # Usar datos sintéticos si APIs fallan
}

# === CONFIGURACIONES DE DATOS SINTÉTICOS ===
SYNTHETIC_CONFIG = {
    "enabled": True,  # Habilitar datos sintéticos por defecto
    "force_synthetic": True,  # Forzar uso de datos sintéticos, ignorar APIs
    "training_period_days": 7,  # Solo 7 días para entrenamiento ultra-rápido
    "prediction_period_days": 1   # 1 día para predicciones
}

# === CONFIGURACIONES DE LOGGING ===
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "weather_model.log",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# === CONFIGURACIONES DE ARCHIVOS ===
MODEL_SAVE_PATH = MODELS_DIR / "weather_model.pt"
SCALER_SAVE_PATH = MODELS_DIR / "scaler.pkl"
FEATURE_COLUMNS_SAVE_PATH = MODELS_DIR / "feature_columns.pkl"

# === CONFIGURACIONES DE CACHÉ ===
CACHE_CONFIG = {
    "postal_coords_cache": POSTAL_DATA_DIR / "postal_coords_cache.pkl",
    "weather_data_cache": WEATHER_DATA_DIR / "weather_cache",
    "max_cache_age_days": 7  # Días antes de refrescar caché
}

# === CONFIGURACIONES DE VISUALIZACIÓN ===
PLOT_CONFIG = {
    "figsize": (12, 8),
    "dpi": 100,
    "style": "seaborn-v0_8",
    "colors": {
        "temperature": "#FF6B6B",
        "precipitation": "#4ECDC4",
        "humidity": "#45B7D1",
        "wind": "#96CEB4",
        "pressure": "#FECA57",
        "clouds": "#FF9FF3"
    }
}
