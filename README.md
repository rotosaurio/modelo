# Sistema de Prediccion Meteorologica para Chihuahua

Sistema completo de inteligencia artificial para predicción meteorológica precisa en códigos postales del estado de Chihuahua, México. Utiliza datos satelitales de NASA, reanálisis de Copernicus y observaciones de estaciones locales para generar pronósticos de alta precisión.

## 🚀 Características Principales

- **Pronóstico de 6 horas**: Basado en 6 horas de datos históricos
- **Intervalos de 15 minutos**: Predicciones detalladas cada 15 minutos
- **Múltiples fuentes de datos**:
  - NASA GPM IMERG (precipitación)
  - ERA5/ERA5-Land de Copernicus (temperatura, humedad, viento, presión, nubosidad)
  - Meteostat (observaciones de estaciones locales)
- **Modelo LSTM Encoder-Decoder**: Arquitectura avanzada para series temporales
- **Visualizaciones completas**: Gráficos y tablas detalladas
- **API simple**: Función `get_weather_forecast(postal_code)` para integración

## 📊 Variables Pronosticadas

- **Temperatura** (C)
- **Precipitacion** (mm)
- 💧 **Humedad relativa** (%)
- 💨 **Velocidad del viento** (m/s)
- 📊 **Dirección del viento** (°)
- 🌪️ **Presión atmosférica** (hPa)
- ☁️ **Cobertura de nubes** (%)

## 🛠️ Instalación

### 1. Clonar el repositorio
```bash
git clone <url-del-repositorio>
cd backspaceaps
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Configurar credenciales de APIs

Crea un archivo `.env` en el directorio del proyecto con tus credenciales:

```bash
# Copernicus CDS API (ERA5)
CDS_API_KEY=tu_api_key_de_cds

# NASA Earthdata (IMERG)
NASA_EARTHDATA_USERNAME=tu_usuario
NASA_EARTHDATA_PASSWORD=tu_contraseña
```

**Obtener credenciales:**

#### Copernicus CDS API:
1. Regístrate en https://cds.climate.copernicus.eu
2. Ve a tu perfil → API Key
3. Copia la API key completa

#### NASA Earthdata:
1. Regístrate en https://urs.earthdata.nasa.gov
2. Ve a "My Account" → "Profile"
3. Copia tu usuario y contraseña

### 4. Verificar configuración
```bash
python test_credentials.py
```

Si todo está OK, verás:
```
EXITO: Todas las credenciales están configuradas correctamente!
```

## 🚀 Uso Rápido

### Configuración inicial
```bash
# Configurar base de datos de códigos postales
python main.py setup
```

### Entrenar modelo
```bash
# Entrenar modelo completo (puede tomar varias horas)
python main.py train
```

### Generar pronóstico
```bash
# Pronóstico para código postal específico
python main.py predict 31125
```

## 📖 API de Python

### Función principal
```python
from predict_weather import get_weather_forecast

# Obtener pronóstico para código postal
forecast = get_weather_forecast("31125")

# Resultado
{
    "postal_code": "31125",
    "coords": [28.648, -106.086],
    "forecast": [
        {
            "time": "+15min",
            "temp": 24.1,
            "rain": 0.0,
            "humidity": 45.2,
            "wind_speed": 3.2,
            "pressure": 1013.5,
            "cloud_cover": 25.0,
            "desc": "Despejado"
        },
        # ... más pasos de 15 minutos hasta 6 horas
    ]
}
```

### Visualizaciones
```python
from predict_weather import get_weather_forecast
from visualization import display_forecast

# Obtener y mostrar pronóstico con gráficos
forecast = get_weather_forecast("31125")
display_forecast(forecast)
```

## 📁 Estructura del Proyecto

```
backspaceaps/
├── config.py                 # Configuraciones y constantes
├── requirements.txt          # Dependencias de Python
├── main.py                   # Script principal
├── predict_weather.py        # API de predicción
├── README.md                 # Esta documentación
│
├── postal_coordinates.py     # Gestión de coordenadas postales
├── data_imerg.py            # Descarga de datos IMERG (NASA)
├── data_era5.py             # Descarga de datos ERA5 (Copernicus)
├── data_meteostat.py        # Descarga de datos estaciones locales
├── data_preprocessing.py    # Preprocesamiento y unificación de datos
├── feature_engineering.py   # Generación de características derivadas
├── dataset_creation.py      # Creación de datasets supervisados
├── weather_model.py         # Arquitectura del modelo LSTM
├── training_pipeline.py     # Pipeline completo de entrenamiento
├── visualization.py         # Gráficos y visualizaciones
│
├── data/                    # Datos descargados
│   ├── postal_codes/        # Coordenadas de códigos postales
│   └── weather/            # Datos climáticos por fuente
├── models/                  # Modelos entrenados y artefactos
├── logs/                    # Logs de ejecución
└── __pycache__/            # Archivos compilados de Python
```

## 🎯 Comandos Disponibles

```bash
# Configuración
python main.py setup                    # Configurar base de datos postal
python main.py status                   # Ver estado del sistema

# Entrenamiento
python main.py train                    # Entrenar modelo
python main.py train --model-type simple # Entrenar modelo simplificado
python main.py train --force-retrain    # Reentrenar forzadamente

# Predicción
python main.py predict 31125            # Pronóstico para código postal
python main.py predict 31125 --no-viz   # Sin visualizaciones

# Utilidades
python main.py list                     # Listar códigos postales disponibles
python main.py list --limit 10          # Limitar lista a 10 códigos
```

## 📈 Arquitectura del Modelo

### LSTM Encoder-Decoder Multisalida
- **Encoder**: Procesa 6 horas de datos históricos (24 pasos de 15 min)
- **Decoder**: Genera predicciones para las próximas 6 horas (24 pasos)
- **Atención**: Mecanismo de atención para foco en datos relevantes
- **Salidas múltiples**: 6 variables climáticas simultáneamente

### Características de Entrada
- Variables climáticas crudas
- Características temporales (hora, día, mes, estación)
- Lags de diferentes períodos (1h, 2h, 3h, 6h, 12h, 24h)
- Estadísticas rolling (media, std, min, max)
- Características específicas del clima (índice de calor, punto de rocío, etc.)

## 🌍 Datos y Fuentes

### Códigos Postales
- **Fuente**: INEGI (Instituto Nacional de Estadística y Geografía)
- **Cobertura**: Estado de Chihuahua completo
- **Actualización**: Datos oficiales más recientes

### Datos Climáticos
- **NASA GPM IMERG**: Precipitación de alta resolución (30 min → 15 min)
- **ERA5 Copernicus**: Variables atmosféricas reanalizadas (1 hora → 15 min)
- **Meteostat**: Observaciones de estaciones meteorológicas locales

### Unificación Temporal
- Todos los datos se interpolan a intervalos de 15 minutos
- Alineación temporal UTC precisa
- Relleno de valores faltantes con interpolación

## 🎨 Visualizaciones

### Gráficos Disponibles
- **Panel completo**: 6 variables en un gráfico con 6 subplots
- **Línea temporal**: Evolución temporal con íconos y descripciones
- **Tabla resumen**: Formato de consola legible

### Ejemplo de Salida en Consola
```
🌤️ PRONÓSTICO METEOROLÓGICO
================================================================================
Código Postal: 31125
Coordenadas: 28.648, -106.086
Generado: 2025-10-04 12:30:00 UTC
================================================================================
Tiempo      | Temp | Lluvia | Humedad | Viento | Presión | Nubes | Descripción
--------------------------------------------------------------------------------
+15min      | 24.1°C| 0.0mm | 45%    | 3.2m/s| 1013hPa| 25%  | Despejado
+30min      | 23.8°C| 0.1mm | 47%    | 3.0m/s| 1012hPa| 30%  | Parcialmente nublado
+45min      | 23.5°C| 0.3mm | 52%    | 2.8m/s| 1011hPa| 45%  | Lluvia ligera
...
+6h        | 21.0°C| 0.0mm | 38%    | 4.1m/s| 1008hPa| 15%  | Despejado
```

## 🔧 Configuración Avanzada

### Parámetros del Modelo
Editar `config.py` para ajustar:
- Ventanas temporal (INPUT_WINDOW_HOURS, OUTPUT_WINDOW_HOURS)
- Arquitectura del modelo (HIDDEN_SIZE, NUM_LAYERS)
- Hiperparámetros de entrenamiento (LEARNING_RATE, BATCH_SIZE)

### Umbrales de Clasificación
```python
WEATHER_THRESHOLDS = {
    "rain_light": 0.2,      # mm/h - Lluvia ligera
    "rain_moderate": 2.5,   # mm/h - Lluvia moderada
    "rain_heavy": 7.6,      # mm/h - Lluvia fuerte
    "cloudy": 60,           # % - Nublado
    "cold": 10,             # °C - Frío
    "hot": 30               # °C - Caluroso
}
```

## 📋 Requisitos del Sistema

- **Python**: 3.8+
- **RAM**: 8GB mínimo, 16GB recomendado
- **Almacenamiento**: 10GB para datos y modelos
- **GPU**: Opcional pero recomendado para entrenamiento
- **Internet**: Conexión estable para descarga de datos

## 🚨 Solución de Problemas

### Error de credenciales
```
Verificar que el archivo .env existe y contiene credenciales válidas:
- CDS_API_KEY=tu_api_key
- NASA_EARTHDATA_USERNAME=tu_usuario
- NASA_EARTHDATA_PASSWORD=tu_contraseña
```

### Sin datos para código postal
```
Ejecutar: python main.py setup
Verificar que el código postal existe en Chihuahua
```

### Error de memoria
```
Reducir BATCH_SIZE en config.py
Usar modelo simple: python main.py train --model-type simple
```

### Modelo no carga
```
Verificar que existe models/weather_model.pt
Reentrenar: python main.py train --force-retrain
```

## 📄 Licencia

Este proyecto está desarrollado para fines de investigación y predicción meteorológica precisa en el estado de Chihuahua.

## 🤝 Contribución

Para mejoras o reportes de bugs, por favor contactar al desarrollador.

## 📞 Soporte

Para soporte técnico o preguntas sobre el sistema, referirse a la documentación del código o contactar al equipo de desarrollo.

---

**Desarrollado con ❤️ para proporcionar predicciones meteorológicas precisas y accesibles para Chihuahua.**
