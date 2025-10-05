"""
Pipeline completo de entrenamiento del modelo de predicción meteorológica
Coordina la creación de datasets, entrenamiento y evaluación del modelo
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    INPUT_STEPS, OUTPUT_STEPS, TRAINING_CONFIG, MODEL_SAVE_PATH,
    SCALER_SAVE_PATH, FEATURE_COLUMNS_SAVE_PATH, LOGS_DIR, SYNTHETIC_CONFIG
)
from postal_coordinates import PostalCoordinatesManager
from dataset_creation import WeatherDatasetCreator
from weather_model import create_model, ModelTrainer
from data_preprocessing import WeatherDataPreprocessor

logger = logging.getLogger(__name__)


class WeatherTrainingPipeline:
    """Pipeline completo para entrenar el modelo de predicción meteorológica"""

    def __init__(self):
        self.coord_manager = PostalCoordinatesManager()
        self.dataset_creator = WeatherDatasetCreator()
        self.preprocessor = WeatherDataPreprocessor()
        self.trainer = None
        self.model = None

        # Configurar logging
        self._setup_logging()

    def _setup_logging(self):
        """Configura el sistema de logging"""
        log_file = LOGS_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        logger.info(f"Logging configurado. Archivo: {log_file}")

    def prepare_training_data(self, postal_codes: Optional[List[str]] = None,
                            start_date: Optional[pd.Timestamp] = None,
                            end_date: Optional[pd.Timestamp] = None) -> Optional[Dict[str, Any]]:
        """
        Prepara los datos de entrenamiento
        Args:
            postal_codes: Lista de códigos postales (None para usar todos disponibles)
            start_date: Fecha de inicio (None para usar datos históricos)
            end_date: Fecha de fin (None para usar hasta hoy)
        Returns:
            Información del dataset creado
        """
        try:
            logger.info("=== PREPARANDO DATOS DE ENTRENAMIENTO ===")

            # Configurar fechas por defecto
            if end_date is None:
                end_date = pd.Timestamp.now()
            if start_date is None:
                # Si datos sintéticos están habilitados, usar período configurado
                if SYNTHETIC_CONFIG["enabled"]:
                    training_days = SYNTHETIC_CONFIG["training_period_days"]
                    start_date = end_date - pd.DateOffset(days=training_days)
                    logger.info(f"Usando datos sintéticos con período de {training_days} días")
                else:
                    # Usar datos de los últimos 2 años para tener suficiente variabilidad
                    start_date = end_date - pd.DateOffset(years=2)

            logger.info(f"Período de entrenamiento: {start_date.date()} - {end_date.date()}")

            # Obtener códigos postales
            if postal_codes is None:
                postal_codes = self._select_training_postal_codes()
                logger.info(f"Seleccionados {len(postal_codes)} códigos postales para entrenamiento")

            # Crear dataset de entrenamiento
            dataset_info = self.dataset_creator.create_training_dataset(
                postal_codes, start_date, end_date
            )

            if dataset_info is None:
                logger.error("Error creando dataset de entrenamiento")
                return None

            # Guardar información del dataset
            self.dataset_creator.save_dataset_info(dataset_info)

            # Mostrar estadísticas del dataset
            self._show_dataset_statistics(dataset_info)

            return dataset_info

        except Exception as e:
            logger.error(f"Error preparando datos de entrenamiento: {e}")
            return None

    def _select_training_postal_codes(self, max_codes: int = 10) -> List[str]:
        """
        Selecciona códigos postales para entrenamiento
        Prioriza códigos postales con buena distribución geográfica
        """
        try:
            # Obtener todos los códigos postales disponibles
            all_codes = self.coord_manager.get_all_postal_codes()

            if len(all_codes) <= max_codes:
                return all_codes

            # Seleccionar códigos postales con buena distribución
            # Estratificar por zona geográfica de Chihuahua
            northern_codes = [cp for cp in all_codes if cp.startswith('31')]  # Ciudad Juárez y norte
            central_codes = [cp for cp in all_codes if cp.startswith('32')]   # Chihuahua capital y centro
            southern_codes = [cp for cp in all_codes if cp.startswith('33')]  # Sur del estado

            selected_codes = []

            # Tomar proporcionalmente de cada zona
            n_north = min(len(northern_codes), max_codes // 3 + 1)
            n_central = min(len(central_codes), max_codes // 3 + 1)
            n_south = min(len(southern_codes), max_codes - n_north - n_central)

            selected_codes.extend(np.random.choice(northern_codes, n_north, replace=False))
            selected_codes.extend(np.random.choice(central_codes, n_central, replace=False))
            if n_south > 0 and southern_codes:
                selected_codes.extend(np.random.choice(southern_codes, n_south, replace=False))

            logger.info(f"Seleccionados {len(selected_codes)} códigos postales estratificados")
            return selected_codes

        except Exception as e:
            logger.error(f"Error seleccionando códigos postales: {e}")
            # Fallback: seleccionar aleatoriamente
            return list(np.random.choice(all_codes, min(max_codes, len(all_codes)), replace=False))

    def _show_dataset_statistics(self, dataset_info: Dict[str, Any]):
        """Muestra estadísticas del dataset creado"""
        try:
            stats = self.dataset_creator.get_data_statistics(dataset_info)

            logger.info("=== ESTADÍSTICAS DEL DATASET ===")
            logger.info(f"Características de entrada: {stats.get('n_features', 0)}")
            logger.info(f"Variables objetivo: {stats.get('n_targets', 0)}")
            logger.info(f"Pasos de entrada: {stats.get('input_steps', INPUT_STEPS)} (6 horas)")
            logger.info(f"Pasos de salida: {stats.get('output_steps', OUTPUT_STEPS)} (6 horas)")

            for split_name, split_stats in stats.items():
                if split_name.endswith('_dataset'):
                    logger.info(f"{split_name.upper()}:")
                    logger.info(f"  Muestras: {split_stats.get('n_samples', 0)}")
                    logger.info(".6f")
                    logger.info(".6f")

        except Exception as e:
            logger.error(f"Error mostrando estadísticas: {e}")

    def train_model(self, dataset_info: Dict[str, Any],
                   model_type: str = 'full') -> Optional[ModelTrainer]:
        """
        Entrena el modelo de predicción
        Args:
            dataset_info: Información del dataset
            model_type: Tipo de modelo ('full' o 'simple')
        Returns:
            Trainer con el modelo entrenado
        """
        try:
            logger.info("=== ENTRENANDO MODELO ===")

            # Crear modelo
            n_features = dataset_info['n_features']
            n_targets = dataset_info['n_targets']

            self.model = create_model(
                input_size=n_features,
                num_targets=n_targets,
                model_type=model_type
            )

            # Crear trainer
            self.trainer = ModelTrainer(
                model=self.model,
                learning_rate=TRAINING_CONFIG['learning_rate']
            )

            # Entrenar modelo
            train_loader = dataset_info['train_loader']
            val_loader = dataset_info['val_loader']

            training_history = self.trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=TRAINING_CONFIG['num_epochs'],
                patience=TRAINING_CONFIG['patience']
            )

            # Guardar modelo final
            self.trainer.save_model()

            # Mostrar resultados de entrenamiento
            self._show_training_results(training_history)

            return self.trainer

        except Exception as e:
            import traceback
            logger.error(f"Error entrenando modelo: {e}")
            logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            return None

    def _show_training_results(self, history: Dict[str, Any]):
        """Muestra resultados del entrenamiento"""
        try:
            logger.info("=== RESULTADOS DEL ENTRENAMIENTO ===")
            logger.info(f"Mejor época: {history.get('best_epoch', 0)}")
            logger.info(".6f")
            logger.info(f"Épocas totales: {len(history.get('train_loss', []))}")

            # Crear gráfico de pérdida
            self._plot_training_history(history)

        except Exception as e:
            logger.error(f"Error mostrando resultados: {e}")

    def _plot_training_history(self, history: Dict[str, Any]):
        """Crea gráfico del historial de entrenamiento"""
        try:
            plt.figure(figsize=(12, 6))

            train_loss = history.get('train_loss', [])
            val_loss = history.get('val_loss', [])
            epochs = range(1, len(train_loss) + 1)

            plt.plot(epochs, train_loss, 'b-', label='Pérdida de entrenamiento', linewidth=2)
            plt.plot(epochs, val_loss, 'r-', label='Pérdida de validación', linewidth=2)

            plt.xlabel('Época', fontsize=12)
            plt.ylabel('Pérdida (MSE)', fontsize=12)
            plt.title('Historial de Entrenamiento del Modelo', fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)

            # Marcar mejor época
            best_epoch = history.get('best_epoch', 0)
            if best_epoch > 0 and best_epoch <= len(val_loss):
                best_loss = val_loss[best_epoch - 1]
                plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7,
                           label=f'Mejor modelo (Época {best_epoch})')
                plt.plot(best_epoch, best_loss, 'go', markersize=8)

            plt.legend()
            plt.tight_layout()

            # Guardar gráfico
            plot_path = LOGS_DIR / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"Gráfico de entrenamiento guardado: {plot_path}")

            plt.close()

        except Exception as e:
            logger.error(f"Error creando gráfico de entrenamiento: {e}")

    def evaluate_model(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evalúa el modelo en el conjunto de test
        Args:
            dataset_info: Información del dataset
        Returns:
            Métricas de evaluación
        """
        try:
            logger.info("=== EVALUANDO MODELO ===")

            if self.trainer is None:
                logger.error("No hay modelo entrenado para evaluar")
                return {}

            test_loader = dataset_info['test_loader']
            test_loss = self.trainer.validate(test_loader)

            # Calcular métricas adicionales
            metrics = self._calculate_metrics(test_loader)

            logger.info("=== MÉTRICAS DE EVALUACIÓN ===")
            logger.info(".6f")
            for var, var_metrics in metrics.items():
                logger.info(f"{var.upper()}:")
                logger.info(".4f")
                logger.info(".4f")
                logger.info(".4f")

            return {
                'test_loss': test_loss,
                'metrics_by_variable': metrics
            }

        except Exception as e:
            logger.error(f"Error evaluando modelo: {e}")
            return {}

    def _calculate_metrics(self, test_loader) -> Dict[str, float]:
        """
        Calcula métricas de evaluación detalladas por variable
        Args:
            test_loader: DataLoader de test
        Returns:
            Diccionario con métricas por variable
        """
        try:
            import torch
            self.trainer.model.eval()

            all_predictions = []
            all_targets = []

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(self.trainer.device)
                    batch_y = batch_y.to(self.trainer.device)

                    predictions = self.trainer.model(batch_X)

                    all_predictions.append(predictions.cpu().numpy())
                    all_targets.append(batch_y.cpu().numpy())

            predictions = np.concatenate(all_predictions, axis=0)
            targets = np.concatenate(all_targets, axis=0)

            # Desnormalizar para métricas absolutas
            predictions_denorm = self.preprocessor.denormalize_data(
                pd.DataFrame(predictions.reshape(-1, predictions.shape[-1]),
                           columns=self.preprocessor.feature_columns[:predictions.shape[-1]])
            ).values.reshape(predictions.shape)

            targets_denorm = self.preprocessor.denormalize_data(
                pd.DataFrame(targets.reshape(-1, targets.shape[-1]),
                           columns=self.preprocessor.feature_columns[:targets.shape[-1]])
            ).values.reshape(targets.shape)

            # Calcular métricas por variable
            target_variables = self.dataset_creator.target_variables
            metrics = {}

            for i, var in enumerate(target_variables):
                pred_var = predictions_denorm[:, :, i].flatten()
                true_var = targets_denorm[:, :, i].flatten()

                # Remover NaN
                valid_mask = ~(np.isnan(pred_var) | np.isnan(true_var))
                pred_var = pred_var[valid_mask]
                true_var = true_var[valid_mask]

                if len(pred_var) == 0:
                    continue

                # Calcular métricas
                mse = np.mean((pred_var - true_var) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(pred_var - true_var))

                # R² score
                ss_res = np.sum((true_var - pred_var) ** 2)
                ss_tot = np.sum((true_var - np.mean(true_var)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                metrics[var] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }

            return metrics

        except Exception as e:
            logger.error(f"Error calculando métricas: {e}")
            return {}

    def save_training_artifacts(self, dataset_info: Dict[str, Any],
                               evaluation_results: Dict[str, Any]):
        """
        Guarda todos los artefactos del entrenamiento
        Args:
            dataset_info: Información del dataset
            evaluation_results: Resultados de evaluación
        """
        try:
            logger.info("=== GUARDANDO ARTEFACTOS DE ENTRENAMIENTO ===")

            # Guardar escalador
            import joblib
            joblib.dump(self.preprocessor.scaler, SCALER_SAVE_PATH)
            logger.info(f"Escalador guardado: {SCALER_SAVE_PATH}")

            # Guardar columnas de características
            with open(FEATURE_COLUMNS_SAVE_PATH, 'w') as f:
                f.write('\n'.join(dataset_info.get('feature_columns', [])))
            logger.info(f"Columnas de features guardadas: {FEATURE_COLUMNS_SAVE_PATH}")

            # Guardar información del dataset
            self.dataset_creator.save_dataset_info(dataset_info)

            logger.info("Artefactos de entrenamiento guardados correctamente")

        except Exception as e:
            logger.error(f"Error guardando artefactos: {e}")

    def run_full_training_pipeline(self, postal_codes: Optional[List[str]] = None,
                                 model_type: str = 'full') -> bool:
        """
        Ejecuta el pipeline completo de entrenamiento
        Args:
            postal_codes: Códigos postales para entrenamiento
            model_type: Tipo de modelo
        Returns:
            True si el entrenamiento fue exitoso
        """
        try:
            logger.info("=== INICIANDO PIPELINE COMPLETO DE ENTRENAMIENTO ===")

            # 1. Preparar datos
            dataset_info = self.prepare_training_data(postal_codes)
            if dataset_info is None:
                return False

            # 2. Entrenar modelo
            trainer = self.train_model(dataset_info, model_type)
            if trainer is None:
                return False

            # 3. Evaluar modelo
            evaluation_results = self.evaluate_model(dataset_info)

            # 4. Guardar artefactos
            self.save_training_artifacts(dataset_info, evaluation_results)

            logger.info("=== PIPELINE DE ENTRENAMIENTO COMPLETADO EXITOSAMENTE ===")
            return True

        except Exception as e:
            logger.error(f"Error en pipeline de entrenamiento: {e}")
            return False
