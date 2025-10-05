"""
Módulo para crear datasets supervisados para predicción de series temporales
Crea ventanas de entrada (6h) y salida (6h) para el modelo de clima
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from config import INPUT_STEPS, OUTPUT_STEPS, TRAINING_CONFIG, MODEL_SAVE_PATH
from data_preprocessing import WeatherDataPreprocessor
from feature_engineering import WeatherFeatureEngineer

logger = logging.getLogger(__name__)


class WeatherDataset(Dataset):
    """Dataset personalizado para datos climáticos de series temporales"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Array de secuencias de entrada (batch_size, input_steps, features)
            y: Array de secuencias de salida (batch_size, output_steps, targets)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class WeatherDatasetCreator:
    """Clase para crear datasets supervisados de predicción climática"""

    def __init__(self):
        self.preprocessor = WeatherDataPreprocessor()
        self.feature_engineer = WeatherFeatureEngineer()
        self.target_variables = ['temp_celsius', 'precipitation_mm', 'relative_humidity_percent',
                               'wind_speed_ms', 'pressure_hpa', 'cloud_cover_percent']

    def create_training_dataset(self, postal_codes: List[str],
                              start_date: pd.Timestamp,
                              end_date: pd.Timestamp) -> Optional[Dict[str, Any]]:
        """
        Crea dataset de entrenamiento recopilando datos de múltiples códigos postales
        Args:
            postal_codes: Lista de códigos postales para entrenar
            start_date: Fecha de inicio del período de entrenamiento
            end_date: Fecha de fin del período de entrenamiento
        Returns:
            Diccionario con datasets de train/val/test o None si falla
        """
        try:
            logger.info(f"Creando dataset de entrenamiento con {len(postal_codes)} códigos postales")

            all_sequences = []

            for i, postal_code in enumerate(postal_codes):
                logger.info(f"Procesando código postal {postal_code} ({i+1}/{len(postal_codes)})")

                # Obtener coordenadas
                from postal_coordinates import PostalCoordinatesManager
                coord_manager = PostalCoordinatesManager()
                coords = coord_manager.get_coordinates(postal_code)

                if not coords:
                    logger.warning(f"No se encontraron coordenadas para {postal_code}")
                    continue

                lat, lon = coords

                # Recopilar datos climáticos
                weather_data = self.preprocessor.collect_weather_data(lat, lon, postal_code,
                                                                     start_date.to_pydatetime(),
                                                                     end_date.to_pydatetime())

                if weather_data is None or weather_data.empty:
                    logger.warning(f"No se pudieron obtener datos para {postal_code}")
                    continue

                # Crear características derivadas
                weather_features = self.feature_engineer.create_all_features(weather_data)

                # Seleccionar características para el modelo
                weather_selected = self.feature_engineer.select_features_for_model(weather_features)

                # Normalizar datos
                weather_normalized = self.preprocessor.normalize_data(weather_selected, fit_scaler=(i == 0))

                # Crear secuencias
                X, y = self.preprocessor.prepare_sequences(
                    weather_normalized,
                    INPUT_STEPS,
                    OUTPUT_STEPS,
                    self.target_variables
                )

                if len(X) > 0 and len(y) > 0:
                    # Limitar número de secuencias por código postal (debe ser divisible por batch_size=2)
                    max_sequences_per_postal = 40  # Divisible por 2
                    if len(X) > max_sequences_per_postal:
                        indices = np.random.choice(len(X), max_sequences_per_postal, replace=False)
                        X = X[indices]
                        y = y[indices]
                    elif len(X) < max_sequences_per_postal:
                        # Si hay menos secuencias, duplicar algunas para llegar al máximo
                        n_needed = max_sequences_per_postal - len(X)
                        if n_needed > 0 and len(X) > 0:
                            indices = np.random.choice(len(X), n_needed, replace=True)
                            X = np.concatenate([X, X[indices]], axis=0)
                            y = np.concatenate([y, y[indices]], axis=0)

                    # Agregar código postal como metadato
                    sequences_data = {
                        'X': X,
                        'y': y,
                        'postal_code': postal_code,
                        'n_sequences': len(X)
                    }
                    all_sequences.append(sequences_data)

                    logger.info(f"Secuencias creadas para {postal_code}: {len(X)}")

            if not all_sequences:
                logger.error("No se pudieron crear secuencias de entrenamiento")
                return None

            # Combinar todas las secuencias
            X_combined = np.concatenate([seq['X'] for seq in all_sequences])
            y_combined = np.concatenate([seq['y'] for seq in all_sequences])

            logger.info(f"Dataset combinado: {len(X_combined)} secuencias totales")

            # Dividir en train/val/test
            train_val_size = 1 - TRAINING_CONFIG['test_split']
            val_size = TRAINING_CONFIG['validation_split'] / train_val_size

            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X_combined, y_combined,
                test_size=TRAINING_CONFIG['test_split'],
                random_state=TRAINING_CONFIG['random_seed'],
                shuffle=True
            )

            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=val_size,
                random_state=TRAINING_CONFIG['random_seed'],
                shuffle=True
            )

            # Crear DataLoaders
            train_dataset = WeatherDataset(X_train, y_train)
            val_dataset = WeatherDataset(X_val, y_val)
            test_dataset = WeatherDataset(X_test, y_test)

            # Configuración optimizada para DataLoaders
            loader_config_base = {
                'batch_size': TRAINING_CONFIG['batch_size'],
                'num_workers': 2,  # Usar workers para mejor rendimiento
                'pin_memory': True,  # Transferencia GPU más rápida
                'persistent_workers': True,  # Mantener workers activos
                'prefetch_factor': 2,  # Pre-cargar batches
            }

            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                drop_last=True,  # Drop last para train (batches uniformes)
                **loader_config_base
            )

            val_loader = DataLoader(
                val_dataset,
                shuffle=False,
                drop_last=False,  # NO drop last para val (usar todas las muestras)
                **loader_config_base
            )

            test_loader = DataLoader(
                test_dataset,
                shuffle=False,
                drop_last=False,  # NO drop last para test (usar todas las muestras)
                **loader_config_base
            )

            dataset_info = {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'test_loader': test_loader,
                'train_dataset': train_dataset,
                'val_dataset': val_dataset,
                'test_dataset': test_dataset,
                'feature_columns': self.preprocessor.feature_columns,
                'target_variables': self.target_variables,
                'input_steps': INPUT_STEPS,
                'output_steps': OUTPUT_STEPS,
                'n_features': X_combined.shape[2],
                'n_targets': y_combined.shape[2],
                'scaler': self.preprocessor.scaler
            }

            logger.info("Dataset de entrenamiento creado exitosamente:")
            logger.info(f"  Train: {len(X_train)} secuencias")
            logger.info(f"  Val: {len(X_val)} secuencias")
            logger.info(f"  Test: {len(X_test)} secuencias")
            logger.info(f"  Features: {X_combined.shape[2]}, Targets: {y_combined.shape[2]}")

            return dataset_info

        except Exception as e:
            logger.error(f"Error creando dataset de entrenamiento: {e}")
            return None

    def create_prediction_dataset(self, lat: float, lon: float, postal_code: str) -> Optional[Dict[str, Any]]:
        """
        Crea dataset para predicción usando datos recientes
        Args:
            lat: Latitud
            lon: Longitud
            postal_code: Código postal
        Returns:
            Diccionario con datos preparados para predicción
        """
        try:
            logger.info(f"Creando dataset de predicción para {postal_code}")

            # Recopilar datos recientes
            recent_data = self.preprocessor.collect_prediction_data(lat, lon, postal_code)

            if recent_data is None or recent_data.empty:
                logger.error("No se pudieron obtener datos recientes para predicción")
                return None

            # Crear características derivadas
            recent_features = self.feature_engineer.create_all_features(recent_data)

            # Seleccionar características para el modelo
            recent_selected = self.feature_engineer.select_features_for_model(recent_features)

            # Normalizar datos (usar escalador existente)
            recent_normalized = self.preprocessor.normalize_data(recent_selected, fit_scaler=False)

            # Verificar que tenemos suficientes datos para una secuencia completa
            if len(recent_normalized) < INPUT_STEPS:
                logger.error(f"Datos insuficientes: {len(recent_normalized)} < {INPUT_STEPS}")
                return None

            # Tomar los últimos INPUT_STEPS registros
            input_sequence = recent_normalized.iloc[-INPUT_STEPS:].values
            input_sequence = input_sequence.reshape(1, INPUT_STEPS, -1)  # (1, input_steps, features)

            # Crear tensor de entrada
            X_pred = torch.tensor(input_sequence, dtype=torch.float32)

            prediction_info = {
                'X_pred': X_pred,
                'raw_data': recent_data,
                'normalized_data': recent_normalized,
                'feature_columns': self.preprocessor.feature_columns,
                'target_variables': self.target_variables,
                'postal_code': postal_code,
                'coords': (lat, lon),
                'timestamp': recent_data.index[-1]
            }

            logger.info(f"Dataset de predicción creado: {X_pred.shape}")
            return prediction_info

        except Exception as e:
            logger.error(f"Error creando dataset de predicción: {e}")
            return None

    def save_dataset_info(self, dataset_info: Dict[str, Any], filepath: Optional[Path] = None):
        """
        Guarda información del dataset para uso posterior
        Args:
            dataset_info: Información del dataset
            filepath: Path donde guardar (opcional)
        """
        try:
            if filepath is None:
                filepath = MODEL_SAVE_PATH.parent / "dataset_info.pkl"

            # Crear diccionario serializable (excluir DataLoaders y Datasets)
            serializable_info = {
                'feature_columns': dataset_info.get('feature_columns', []),
                'target_variables': dataset_info.get('target_variables', []),
                'input_steps': dataset_info.get('input_steps', INPUT_STEPS),
                'output_steps': dataset_info.get('output_steps', OUTPUT_STEPS),
                'n_features': dataset_info.get('n_features', 0),
                'n_targets': dataset_info.get('n_targets', 0),
                'scaler': dataset_info.get('scaler', None)
            }

            with open(filepath, 'wb') as f:
                pickle.dump(serializable_info, f)

            logger.info(f"Información del dataset guardada: {filepath}")

        except Exception as e:
            logger.error(f"Error guardando información del dataset: {e}")

    def load_dataset_info(self, filepath: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """
        Carga información del dataset desde archivo
        Args:
            filepath: Path del archivo (opcional)
        Returns:
            Información del dataset o None si falla
        """
        try:
            if filepath is None:
                filepath = MODEL_SAVE_PATH.parent / "dataset_info.pkl"

            if not filepath.exists():
                logger.warning(f"Archivo de información del dataset no encontrado: {filepath}")
                return None

            with open(filepath, 'rb') as f:
                dataset_info = pickle.load(f)

            # Restaurar escalador en preprocessor
            if 'scaler' in dataset_info:
                self.preprocessor.scaler = dataset_info['scaler']
                self.preprocessor.feature_columns = dataset_info.get('feature_columns', [])

            logger.info(f"Información del dataset cargada: {filepath}")
            return dataset_info

        except Exception as e:
            logger.error(f"Error cargando información del dataset: {e}")
            return None

    def get_data_statistics(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula estadísticas descriptivas del dataset
        Args:
            dataset_info: Información del dataset
        Returns:
            Diccionario con estadísticas
        """
        try:
            stats = {}

            # Estadísticas de datasets
            for split_name in ['train_dataset', 'val_dataset', 'test_dataset']:
                if split_name in dataset_info:
                    dataset = dataset_info[split_name]
                    X, y = dataset.X.numpy(), dataset.y.numpy()

                    stats[split_name] = {
                        'n_samples': len(dataset),
                        'X_shape': X.shape,
                        'y_shape': y.shape,
                        'X_mean': float(X.mean()),
                        'X_std': float(X.std()),
                        'y_mean': float(y.mean()),
                        'y_std': float(y.std())
                    }

            # Estadísticas por variable objetivo
            if 'target_variables' in dataset_info:
                target_vars = dataset_info['target_variables']
                stats['target_variables'] = target_vars
                stats['n_targets'] = len(target_vars)

            # Información general
            stats['input_steps'] = dataset_info.get('input_steps', INPUT_STEPS)
            stats['output_steps'] = dataset_info.get('output_steps', OUTPUT_STEPS)
            stats['n_features'] = dataset_info.get('n_features', 0)

            return stats

        except Exception as e:
            logger.error(f"Error calculando estadísticas del dataset: {e}")
            return {}
