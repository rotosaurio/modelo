"""
Script principal del sistema de predicción meteorológica para Chihuahua
Coordina todas las fases: descarga de datos, entrenamiento y predicción
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from config import LOGS_DIR
from postal_coordinates import PostalCoordinatesManager
from training_pipeline import WeatherTrainingPipeline
from predict_weather import WeatherPredictor
from visualization import WeatherVisualizer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f"main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class WeatherPredictionSystem:
    """Sistema completo de predicción meteorológica"""

    def __init__(self):
        self.coord_manager = PostalCoordinatesManager()
        self.training_pipeline = WeatherTrainingPipeline()
        self.predictor = WeatherPredictor()
        self.visualizer = WeatherVisualizer()

    def setup_postal_data(self):
        """Configura la base de datos de coordenadas postales"""
        logger.info("CONFIGURANDO BASE DE DATOS POSTAL")

        if not self.coord_manager.postal_coords:
            logger.info("Descargando datos de códigos postales de Chihuahua...")
            success = self.coord_manager.build_postal_database()

            if success:
                n_codes = len(self.coord_manager.postal_coords)
                logger.info(f"Base de datos postal configurada con {n_codes} codigos postales")
                return True
            else:
                logger.error("Error configurando base de datos postal")
                return False
        else:
            n_codes = len(self.coord_manager.postal_coords)
            logger.info(f"Base de datos postal ya configurada ({n_codes} codigos postales)")
            return True

    def train_model(self, postal_codes: Optional[List[str]] = None,
                   model_type: str = 'full', force_retrain: bool = False):
        """
        Entrena el modelo de predicción
        Args:
            postal_codes: Lista específica de códigos postales para entrenamiento
            model_type: Tipo de modelo ('full' o 'simple')
            force_retrain: Si True, reentrena aunque exista modelo
        """
        logger.info("ENTRENANDO MODELO DE PREDICCION")

        # Verificar si ya existe modelo entrenado
        from config import MODEL_SAVE_PATH
        if MODEL_SAVE_PATH.exists() and not force_retrain:
            logger.info("Modelo ya entrenado encontrado. Use --force-retrain para reentrenar.")
            return True

        # Verificar datos postales
        if not self.setup_postal_data():
            return False

        # Ejecutar pipeline de entrenamiento
        success = self.training_pipeline.run_full_training_pipeline(
            postal_codes=postal_codes,
            model_type=model_type
        )

        if success:
            logger.info("Modelo entrenado exitosamente")
            return True
        else:
            logger.error("Error entrenando modelo")
            return False

    def predict_weather(self, postal_code: str, show_visualization: bool = True):
        """
        Genera pronóstico para un código postal específico
        Args:
            postal_code: Código postal de Chihuahua
            show_visualization: Si mostrar gráficos y visualizaciones
        """
        logger.info(f"GENERANDO PRONOSTICO PARA {postal_code}")

        # Verificar código postal
        coords = self.coord_manager.get_coordinates(postal_code)
        if not coords:
            logger.error(f"Codigo postal {postal_code} no encontrado en Chihuahua")
            return None

        logger.info(f"Coordenadas encontradas: {coords[0]:.3f}, {coords[1]:.3f}")

        # Generar pronóstico
        forecast = self.predictor.get_weather_forecast(postal_code)

        if "error" in forecast:
            logger.error(f"Error en pronostico: {forecast['error']}")
            return None

        # Mostrar pronóstico en consola
        self.predictor.display_forecast_console(forecast)

        # Mostrar visualizaciones si se solicita
        if show_visualization:
            logger.info("Generando visualizaciones...")
            self.visualizer.display_forecast_summary(forecast)

        logger.info("Pronostico generado exitosamente")
        return forecast

    def validate_postal_code(self, postal_code: str) -> bool:
        """
        Valida si un código postal existe en Chihuahua
        Args:
            postal_code: Código postal a validar
        Returns:
            True si existe
        """
        coords = self.coord_manager.get_coordinates(postal_code)
        return coords is not None

    def list_available_postal_codes(self, limit: Optional[int] = None):
        """
        Lista códigos postales disponibles
        Args:
            limit: Número máximo a mostrar
        """
        codes = self.coord_manager.get_all_postal_codes()

        if not codes:
            logger.warning("No hay códigos postales disponibles. Configure la base de datos primero.")
            return

        print(f"Codigos postales disponibles en Chihuahua: {len(codes)}")
        print("-" * 50)

        codes_to_show = codes[:limit] if limit else codes
        for i, code in enumerate(codes_to_show, 1):
            coords = self.coord_manager.get_coordinates(code)
            if coords:
                print(f"{i:3}. {code} - Lat: {coords[0]:>7.3f}, Lon: {coords[1]:>8.3f}")

        if limit and len(codes) > limit:
            print(f"... y {len(codes) - limit} más")

    def show_system_status(self):
        """Muestra el estado del sistema"""
        print("SISTEMA DE PREDICCION METEOROLOGICA - CHIHUAHUA")
        print("=" * 60)

        # Estado de base de datos postal
        n_postal_codes = len(self.coord_manager.postal_coords)
        postal_status = "Configurada" if n_postal_codes > 0 else "No configurada"
        print(f"Base de datos postal: {postal_status} ({n_postal_codes} codigos)")

        # Estado del modelo
        from config import MODEL_SAVE_PATH, SCALER_SAVE_PATH
        model_exists = MODEL_SAVE_PATH.exists()
        scaler_exists = SCALER_SAVE_PATH.exists()
        model_status = "Entrenado" if model_exists and scaler_exists else "No entrenado"
        print(f"Modelo de prediccion: {model_status}")

        # Información del sistema
        print(f"Directorio de logs: {LOGS_DIR}")
        print(f"Directorio de modelos: {MODEL_SAVE_PATH.parent}")

        print("\nCAPACIDADES:")
        print("• Pronostico de 6 horas basado en 6 horas de datos historicos")
        print("• Intervalos de 15 minutos")
        print("• Variables: temperatura, precipitacion, humedad, viento, presion, nubosidad")
        print("• Datos de NASA IMERG, ERA5 Copernicus y estaciones locales")

        print("\nUSO RAPIDO:")
        print("python main.py predict 31125                    # Pronostico para codigo postal")
        print("python main.py train                            # Entrenar modelo")
        print("python main.py setup                            # Configurar base de datos")
        print("python main.py list                             # Listar códigos postales")


def main():
    """Función principal del programa"""
    parser = argparse.ArgumentParser(
        description="Sistema de predicción meteorológica para Chihuahua",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py setup                    # Configurar base de datos postal
  python main.py train                    # Entrenar modelo
  python main.py predict 31125            # Pronóstico para Chihuahua capital
  python main.py predict 31000 --no-viz   # Pronóstico sin visualizaciones
  python main.py list                     # Listar códigos postales disponibles
  python main.py status                   # Ver estado del sistema
        """
    )

    parser.add_argument('command', choices=['setup', 'train', 'predict', 'list', 'status'],
                       help='Comando a ejecutar')

    parser.add_argument('postal_code', nargs='?', help='Código postal para predicción')

    parser.add_argument('--model-type', choices=['full', 'simple'], default='full',
                       help='Tipo de modelo a usar (default: full)')

    parser.add_argument('--force-retrain', action='store_true',
                       help='Forzar reentrenamiento del modelo')

    parser.add_argument('--no-viz', action='store_true',
                       help='No mostrar visualizaciones en predicción')

    parser.add_argument('--limit', type=int, default=20,
                       help='Límite para listar códigos postales')

    args = parser.parse_args()

    # Crear sistema
    system = WeatherPredictionSystem()

    try:
        if args.command == 'setup':
            # Configurar base de datos postal
            success = system.setup_postal_data()
            if success:
                print("Base de datos postal configurada correctamente")
            else:
                print("Error configurando base de datos postal")
                sys.exit(1)

        elif args.command == 'train':
            # Entrenar modelo
            success = system.train_model(
                model_type=args.model_type,
                force_retrain=args.force_retrain
            )
            if success:
                print("Modelo entrenado correctamente")
            else:
                print("Error entrenando modelo")
                sys.exit(1)

        elif args.command == 'predict':
            # Generar pronóstico
            if not args.postal_code:
                print("Debe especificar un codigo postal para prediccion")
                print("Ejemplo: python main.py predict 31125")
                sys.exit(1)

            # Validar código postal
            if not system.validate_postal_code(args.postal_code):
                print(f"Codigo postal {args.postal_code} no encontrado en Chihuahua")
                print("Use 'python main.py list' para ver codigos disponibles")
                sys.exit(1)

            forecast = system.predict_weather(
                args.postal_code,
                show_visualization=not args.no_viz
            )

            if forecast is None:
                sys.exit(1)

        elif args.command == 'list':
            # Listar códigos postales
            system.list_available_postal_codes(limit=args.limit)

        elif args.command == 'status':
            # Mostrar estado del sistema
            system.show_system_status()

    except KeyboardInterrupt:
        print("\nOperacion cancelada por el usuario")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Error ejecutando comando {args.command}: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
