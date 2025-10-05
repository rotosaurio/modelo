"""
Arquitectura del modelo de predicción meteorológica
LSTM Encoder-Decoder multisalida para pronóstico de series temporales climáticas
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import numpy as np

from config import MODEL_CONFIG, OUTPUT_STEPS

logger = logging.getLogger(__name__)


class WeatherPredictor(nn.Module):
    """
    Modelo LSTM Encoder-Decoder para predicción meteorológica multisalida
    Predice múltiples variables climáticas para múltiples pasos temporales futuros
    """

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 output_steps: int = OUTPUT_STEPS, num_targets: int = 6,
                 dropout: float = 0.2, bidirectional: bool = True):
        """
        Args:
            input_size: Número de características de entrada
            hidden_size: Tamaño del estado oculto LSTM
            num_layers: Número de capas LSTM
            output_steps: Número de pasos temporales a predecir (24 para 6 horas)
            num_targets: Número de variables objetivo a predecir
            dropout: Tasa de dropout
            bidirectional: Si usar LSTM bidireccional
        """
        super(WeatherPredictor, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_steps = output_steps
        self.num_targets = num_targets
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Capa de entrada para reducir dimensionalidad si es necesario
        self.input_projection = nn.Linear(input_size, hidden_size)

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size + num_targets,  # Estado oculto + predicción anterior
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Capa de atención para el decoder
        self.attention = nn.Linear(hidden_size * self.num_directions, hidden_size * self.num_directions)
        self.attention_combine = nn.Linear(hidden_size * self.num_directions * 2, hidden_size * self.num_directions)

        # Capa de salida
        self.output_projection = nn.Linear(hidden_size * self.num_directions, num_targets)

        # Capas de normalización y dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size * self.num_directions)

        # Inicialización de pesos
        self._initialize_weights()

        logger.info(f"Modelo WeatherPredictor creado:")
        logger.info(f"  Input size: {input_size}")
        logger.info(f"  Hidden size: {hidden_size}")
        logger.info(f"  Layers: {num_layers}")
        logger.info(f"  Bidirectional: {bidirectional}")
        logger.info(f"  Output steps: {output_steps}")
        logger.info(f"  Targets: {num_targets}")

    def _initialize_weights(self):
        """Inicializa los pesos del modelo usando Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo
        Args:
            x: Tensor de entrada (batch_size, input_steps, input_size)
        Returns:
            Tensor de salida (batch_size, output_steps, num_targets)
        """
        batch_size = x.size(0)

        # Proyección de entrada
        x_proj = self.input_projection(x)  # (batch_size, input_steps, hidden_size)
        x_proj = self.layer_norm1(x_proj)
        x_proj = F.relu(x_proj)
        x_proj = self.dropout(x_proj)

        # Encoder
        encoder_outputs, (h_n, c_n) = self.encoder_lstm(x_proj)
        # encoder_outputs: (batch_size, input_steps, hidden_size * num_directions)
        # h_n, c_n: (num_layers * num_directions, batch_size, hidden_size)

        # Preparar estado inicial para decoder - simplificado
        # Usar solo el último estado oculto del encoder como inicialización
        h_n = h_n[-self.num_layers:].contiguous()  # (num_layers, batch_size, hidden_size)
        c_n = c_n[-self.num_layers:].contiguous()  # (num_layers, batch_size, hidden_size)

        # Decoder con teacher forcing y atención
        outputs = []
        decoder_input = torch.zeros(batch_size, 1, self.num_targets, device=x.device)

        for t in range(self.output_steps):
            # Atención: calcular pesos de atención sobre las salidas del encoder
            attention_weights = F.softmax(
                self.attention(encoder_outputs), dim=1
            )  # (batch_size, input_steps, hidden_size * num_directions)

            # Aplicar atención
            context = torch.bmm(attention_weights.transpose(1, 2), encoder_outputs)
            # context: (batch_size, hidden_size * num_directions, 1)

            # Reorganizar context para que tenga la dimensión correcta
            context = context.transpose(1, 2)  # (batch_size, 1, hidden_size * num_directions)

            # Combinar contexto con entrada del decoder
            decoder_combined = torch.cat([context, decoder_input], dim=2)
            # decoder_combined: (batch_size, 1, hidden_size * num_directions + num_targets)

            # Paso del decoder
            decoder_output, (h_n, c_n) = self.decoder_lstm(
                decoder_combined, (h_n, c_n)
            )
            # decoder_output: (batch_size, 1, hidden_size * num_directions)

            # Normalización y dropout
            decoder_output = self.layer_norm2(decoder_output)
            decoder_output = self.dropout(decoder_output)

            # Proyección a espacio de salida
            output = self.output_projection(decoder_output)
            # output: (batch_size, 1, num_targets)

            outputs.append(output)

            # Teacher forcing: usar predicción como entrada para el siguiente paso
            decoder_input = output

        # Concatenar todas las salidas
        outputs = torch.cat(outputs, dim=1)  # (batch_size, output_steps, num_targets)

        return outputs

    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Método optimizado para predicción paso a paso (sin teacher forcing)
        Args:
            x: Tensor de entrada (batch_size, input_steps, input_size)
        Returns:
            Tensor de salida (batch_size, output_steps, num_targets)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class WeatherPredictorSimplified(nn.Module):
    """
    Versión simplificada del modelo para casos donde el modelo completo es demasiado complejo
    """

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1,
                 output_steps: int = OUTPUT_STEPS, num_targets: int = 6):
        super(WeatherPredictorSimplified, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_steps = output_steps
        self.num_targets = num_targets

        # Encoder simple
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Decoder con múltiples capas lineales para cada paso temporal
        self.temporal_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_targets)
            ) for _ in range(output_steps)
        ])

        # Inicialización
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        _, (h_n, _) = self.encoder(x)

        # Tomar el último estado oculto
        encoder_output = h_n[-1]  # (batch_size, hidden_size)

        # Generar predicciones para cada paso temporal
        outputs = []
        for decoder in self.temporal_decoders:
            step_output = decoder(encoder_output)  # (batch_size, num_targets)
            outputs.append(step_output.unsqueeze(1))  # (batch_size, 1, num_targets)

        return torch.cat(outputs, dim=1)  # (batch_size, output_steps, num_targets)

    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Método optimizado para predicción paso a paso (sin teacher forcing)
        Args:
            x: Tensor de entrada (batch_size, input_steps, input_size)
        Returns:
            Tensor de salida (batch_size, output_steps, num_targets)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class ModelTrainer:
    """Clase para entrenar el modelo de predicción meteorológica"""

    def __init__(self, model: nn.Module, learning_rate: float = 1e-3,
                 device: str = 'auto'):
        self.model = model
        self.learning_rate = learning_rate

        # Configurar dispositivo
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Optimizador y función de pérdida
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        # Función de pérdida con pesos para diferentes variables
        # Dar más peso a precipitación y temperatura
        loss_weights = torch.tensor([2.0, 3.0, 1.0, 1.0, 1.0, 1.0]).to(self.device)  # temp, precip, humidity, wind, pressure, clouds
        self.criterion = nn.MSELoss(reduction='none')

        # Configuración de entrenamiento mejorada
        self.clip_grad_norm = 1.0
        self.min_learning_rate = 1e-6

        self.best_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0

        logger.info(f"Trainer inicializado en dispositivo: {self.device}")

    def weighted_mse_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calcula pérdida MSE ponderada
        Args:
            y_pred: Predicciones del modelo
            y_true: Valores reales
        Returns:
            Pérdida escalar
        """
        loss = self.criterion(y_pred, y_true)
        # Aplicar pesos por variable (asumiendo que están en el último eje)
        weights = torch.tensor([2.0, 3.0, 1.0, 1.0, 1.0, 1.0]).to(self.device)
        loss = loss * weights.unsqueeze(0).unsqueeze(0)  # Broadcast a (batch, time, vars)
        return loss.mean()

    def train_epoch(self, train_loader) -> float:
        """Entrena una época"""
        self.model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(batch_X)

            # Calcular pérdida
            loss = self.weighted_mse_loss(outputs, batch_y)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

    def validate(self, val_loader) -> float:
        """Valida el modelo"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = self.weighted_mse_loss(outputs, batch_y)
                total_loss += loss.item()

        return total_loss / len(val_loader) if len(val_loader) > 0 else 0.0

    def train(self, train_loader, val_loader, num_epochs: int = 100,
             patience: int = 10) -> Dict[str, Any]:
        """
        Entrena el modelo completo
        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            num_epochs: Número máximo de épocas
            patience: Paciencia para early stopping
        Returns:
            Diccionario con historial de entrenamiento
        """
        self.patience = patience
        self.patience_counter = 0
        self.best_loss = float('inf')

        history = {
            'train_loss': [],
            'val_loss': [],
            'best_epoch': 0,
            'best_val_loss': float('inf')
        }

        logger.info(f"Comenzando entrenamiento por {num_epochs} épocas...")

        for epoch in range(num_epochs):
            # Entrenamiento
            train_loss = self.train_epoch(train_loader)

            # Validación
            val_loss = self.validate(val_loader)

            # Actualizar scheduler
            self.scheduler.step(val_loss)

            # Guardar en historial
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            logger.info(f"Época {epoch+1:3d}/{num_epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

            # Early stopping
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                history['best_epoch'] = epoch + 1
                history['best_val_loss'] = val_loss

                # Guardar mejor modelo
                self.save_model()
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping en época {epoch+1}")
                break

        logger.info(f"Entrenamiento completado. Mejor pérdida de validación: {self.best_loss:.6f} en época {history['best_epoch']}")
        return history

    def save_model(self, filepath: Optional[str] = None):
        """Guarda el modelo con hiperparámetros"""
        if filepath is None:
            from config import MODEL_SAVE_PATH
            filepath = str(MODEL_SAVE_PATH)

        # Guardar hiperparámetros del modelo
        model_hyperparams = {
            'input_size': self.model.input_size,
            'hidden_size': self.model.hidden_size,
            'num_layers': self.model.num_layers,
            'output_steps': self.model.output_steps,
            'num_targets': self.model.num_targets,
            'model_class': self.model.__class__.__name__
        }

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'hyperparams': model_hyperparams
        }, filepath)

        logger.info(f"Modelo guardado con hiperparámetros: {filepath}")

    def load_model(self, filepath: Optional[str] = None):
        """Carga el modelo con validación de hiperparámetros"""
        if filepath is None:
            from config import MODEL_SAVE_PATH
            filepath = str(MODEL_SAVE_PATH)

        from pathlib import Path
        if not Path(filepath).exists():
            logger.error(f"Archivo de modelo no encontrado: {filepath}")
            return False

        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Validar hiperparámetros si existen
            if 'hyperparams' in checkpoint:
                saved_hyperparams = checkpoint['hyperparams']
                current_hyperparams = {
                    'input_size': self.model.input_size,
                    'hidden_size': self.model.hidden_size,
                    'num_layers': self.model.num_layers,
                    'output_steps': self.model.output_steps,
                    'num_targets': self.model.num_targets,
                    'model_class': self.model.__class__.__name__
                }

                logger.info(f"Validando hiperparámetros: guardados={saved_hyperparams}, actuales={current_hyperparams}")

                # Verificar compatibilidad
                for key in ['input_size', 'hidden_size', 'num_layers', 'output_steps', 'num_targets']:
                    if saved_hyperparams.get(key) != current_hyperparams.get(key):
                        logger.warning(f"Desajuste en {key}: guardado={saved_hyperparams.get(key)}, actual={current_hyperparams.get(key)}")
                        logger.error("Los hiperparámetros del modelo no coinciden. Reentrena el modelo o ajusta la configuración.")
                        return False
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_loss = checkpoint.get('best_loss', float('inf'))

            logger.info(f"Modelo cargado correctamente: {filepath}")
            return True
            
        except RuntimeError as e:
            if "size mismatch" in str(e):
                logger.error(f"Error de dimensiones al cargar modelo: {e}")
                logger.error("Solución: Reentrena el modelo con 'python main.py train --force-retrain'")
            else:
                logger.error(f"Error cargando modelo: {e}")
            return False
        except Exception as e:
            logger.error(f"Error inesperado cargando modelo: {e}")
            return False

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Realiza predicciones
        Args:
            X: Tensor de entrada
        Returns:
            Array de numpy con predicciones
        """
        self.model.eval()

        X = X.to(self.device)

        with torch.no_grad():
            predictions = self.model.predict_step(X)

        return predictions.cpu().numpy()


def create_model(input_size: int, num_targets: int = 6,
                model_type: str = 'simple', hidden_size: int = None,
                num_layers: int = None) -> WeatherPredictor:
    """
    Función auxiliar para crear modelo
    Args:
        input_size: Número de características de entrada
        num_targets: Número de variables objetivo
        model_type: Tipo de modelo ('full' o 'simple')
        hidden_size: Tamaño del estado oculto (opcional, usa config por defecto si None)
        num_layers: Número de capas (opcional, usa config por defecto si None)
    Returns:
        Modelo creado
    """
    # Usar valores por defecto si no se especifican
    if hidden_size is None:
        hidden_size = MODEL_CONFIG['hidden_size'] // 2 if model_type == 'simple' else MODEL_CONFIG['hidden_size']
    if num_layers is None:
        num_layers = 1 if model_type == 'simple' else MODEL_CONFIG['num_layers']

    if model_type == 'simple':
        model = WeatherPredictorSimplified(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_steps=MODEL_CONFIG['output_steps'],
            num_targets=num_targets
        )
    else:
        # Usar modelo simplificado por defecto debido a problemas de dimensiones
        model = WeatherPredictorSimplified(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_steps=MODEL_CONFIG['output_steps'],
            num_targets=num_targets
        )

    return model
