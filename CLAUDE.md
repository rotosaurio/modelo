# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Weather prediction system for Chihuahua, Mexico postal codes using LSTM Encoder-Decoder architecture. Integrates satellite data from NASA GPM IMERG, ERA5/ERA5-Land reanalysis from Copernicus, and local station observations from Meteostat to generate 6-hour weather forecasts at 15-minute intervals.

**Key Prediction Variables:** Temperature, precipitation, humidity, wind speed/direction, atmospheric pressure, cloud cover

## Common Commands

### Setup & Installation
```bash
pip install -r requirements.txt
python main.py setup              # Configure postal code database
python test_credentials.py        # Verify API credentials
```

### Training
```bash
python main.py train                    # Train full model
python main.py train --model-type simple # Train simplified model
python main.py train --force-retrain    # Force retrain existing model
```

### Prediction
```bash
python main.py predict 31125            # Generate forecast for postal code
python main.py predict 31125 --no-viz   # Forecast without visualizations
```

### Utilities
```bash
python main.py status                   # Show system status
python main.py list                     # List available postal codes
python main.py list --limit 10          # Limit list to 10 codes
```

### Testing
```bash
python test_credentials.py              # Test API credentials
python test_complete_system.py          # Full system test
python test_prediction.py               # Test prediction functionality
```

## Architecture & Data Flow

### Pipeline Stages
1. **Postal Coordinates** (`postal_coordinates.py`) - Manages Chihuahua postal code coordinates from INEGI data
2. **Data Collection** - Downloads weather data from multiple sources:
   - `data_imerg.py` - NASA GPM IMERG precipitation data (30min → 15min interpolation)
   - `data_era5.py` - ERA5 reanalysis atmospheric variables (1hr → 15min interpolation)
   - `data_meteostat.py` - Local weather station observations
   - `data_synthetic.py` - Synthetic data generator for development/fallback
3. **Preprocessing** (`data_preprocessing.py`) - Unifies all data sources to 15-minute intervals, handles missing values
4. **Feature Engineering** (`feature_engineering.py`) - Creates temporal features, lags, rolling statistics, and derived weather features
5. **Dataset Creation** (`dataset_creation.py`) - Builds supervised learning datasets with input/output windows
6. **Model Training** (`training_pipeline.py`, `weather_model.py`) - LSTM Encoder-Decoder with attention mechanism
7. **Prediction** (`predict_weather.py`) - Main API for generating forecasts
8. **Visualization** (`visualization.py`) - Plotting and display of forecast results

### Model Architecture
- **Encoder:** Processes 6 hours (24 steps of 15min) of historical data
- **Decoder:** Generates predictions for next 6 hours (24 steps)
- **Attention:** Focus mechanism for relevant historical patterns
- **Multi-output:** Simultaneous prediction of 6 weather variables
- Two variants: `WeatherPredictor` (full) and `WeatherPredictorSimplified` (lightweight)

### Key Configuration
All settings in `config.py`:
- `INPUT_WINDOW_HOURS` / `OUTPUT_WINDOW_HOURS` - Temporal windows (default: 6h each)
- `TIME_STEP_MINUTES` - Prediction interval (default: 15min)
- `MODEL_CONFIG` - Architecture hyperparameters
- `TRAINING_CONFIG` - Learning rate, batch size, epochs, validation splits
- `SYNTHETIC_CONFIG` - Controls synthetic data usage (enabled by default for fast development)

### Data Storage
```
data/
  postal_codes/        # INEGI postal code coordinates
  weather/            # Downloaded weather data by source
models/               # Trained model artifacts (.pt, .pkl)
logs/                 # Execution logs
```

## API Credentials

Required environment variables in `.env`:
```bash
CDS_API_KEY=your_copernicus_api_key              # From https://cds.climate.copernicus.eu
NASA_EARTHDATA_USERNAME=your_username            # From https://urs.earthdata.nasa.gov
NASA_EARTHDATA_PASSWORD=your_password
```

The system has fallback to synthetic data if APIs fail or when `SYNTHETIC_CONFIG["force_synthetic"] = True` in `config.py`.

## Main Python API

```python
from predict_weather import get_weather_forecast

# Generate forecast for postal code
forecast = get_weather_forecast("31125")

# Returns dict with:
# {
#   "postal_code": "31125",
#   "coords": [lat, lon],
#   "forecast": [
#     {"time": "+15min", "temp": 24.1, "rain": 0.0, ...},
#     ...
#   ]
# }
```

## Important Implementation Details

### Temporal Processing
- All data sources are interpolated to 15-minute intervals regardless of native resolution (IMERG 30min, ERA5 1hr)
- Input window: 24 steps × 15min = 6 hours
- Output window: 24 steps × 15min = 6 hours
- Feature engineering creates lags at 1h, 2h, 3h, 6h, 12h, 24h intervals

### Model State Management
Three critical files for predictions:
- `models/weather_model.pt` - PyTorch model checkpoint with hyperparameters
- `models/scaler.pkl` - Sklearn scaler for feature normalization
- `models/feature_columns.pkl` - Column order for feature consistency

### Synthetic Data Mode
When `SYNTHETIC_CONFIG["force_synthetic"] = True`:
- Training uses only 1-7 days of synthetic data for rapid development
- Skips API calls entirely
- Useful for testing without credentials or during development

### Weather Classification
Thresholds in `WEATHER_THRESHOLDS` (config.py) determine descriptive text:
- Rain: light (>0.2mm/h), moderate (>2.5mm/h), heavy (>7.6mm/h)
- Temperature: cold (<10°C), hot (>30°C)
- Cloud cover: cloudy (>60%)
- Wind: windy (>10m/s)

## Error Handling & Troubleshooting

### No Model Found
```bash
python main.py train                     # Train new model
```

### Missing Credentials
```bash
python test_credentials.py               # Verify .env configuration
```

### Out of Memory
Edit `config.py`:
- Reduce `TRAINING_CONFIG["batch_size"]` (default: 8)
- Use `--model-type simple` for lighter model

### No Data for Postal Code
```bash
python main.py setup                     # Rebuild postal database
python main.py list                      # Verify code exists
```

## Development Notes

- The system is designed for Chihuahua state only (INEGI code: 08)
- Supports both CPU and GPU (CUDA) training - automatically detected
- Model checkpointing includes full hyperparameters for reproducibility
- All timestamps are UTC-aligned during preprocessing
- Weather thresholds are configurable for different climate regions
