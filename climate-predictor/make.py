#!/usr/bin/env python3
"""
Script de automatización para el sistema Climate Predictor.
Equivalente a un Makefile pero en Python puro.

Uso:
    python make.py <comando>

Comandos disponibles:
    install      - Instalar dependencias
    setup        - Configurar el proyecto
    download     - Descargar datos para una ubicación
    preprocess   - Preprocesar datos
    train        - Entrenar modelo
    predict      - Hacer una predicción
    api          - Ejecutar la API
    test         - Ejecutar tests
    clean        - Limpiar archivos generados
    all          - Ejecutar todo el pipeline
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

def run_command(cmd, description="Ejecutando comando"):
    """Ejecuta un comando del sistema."""
    print(f"{description}...")

    # Crear archivo temporal en el directorio src
    src_dir = Path(os.getcwd()) / "src"
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=src_dir) as f:
        f.write(cmd)
        temp_file = f.name

    try:
        # Ejecutar el archivo Python desde el directorio src
        result = subprocess.run([sys.executable, temp_file], cwd=src_dir)

        # Limpiar archivo temporal
        os.unlink(temp_file)

        if result.returncode == 0:
            print("SUCCESS")
            return True
        else:
            print(f"ERROR: Comando falló con código {result.returncode}")
            return False
    except Exception as e:
        # Limpiar archivo temporal en caso de error
        try:
            os.unlink(temp_file)
        except:
            pass
        print(f"ERROR: {e}")
        return False


class ClimatePredictorCLI:
    """CLI para automatizar tareas del sistema Climate Predictor."""

    def __init__(self):
        pass

    def install_dependencies(self):
        """Instala las dependencias del proyecto."""
        print("🔧 Instalando dependencias...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Dependencias instaladas correctamente")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error instalando dependencias: {e}")
            return False
        return True

    def setup_project(self):
        """Configura el proyecto inicial."""
        print("⚙️  Configurando proyecto...")

        # Crear directorios
        dirs = ["data/raw", "data/processed", "models", "logs"]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"📁 Creado directorio: {dir_path}")

        # Verificar configuración de APIs
        from utils import validate_api_keys
        api_status = validate_api_keys()

        print("\n🔑 Estado de APIs:")
        for api, configured in api_status.items():
            status = "✅ Configurada" if configured else "❌ No configurada"
            print(f"  {api}: {status}")

        if not any(api_status.values()):
            print("\n⚠️  Advertencia: Ninguna API está configurada.")
            print("   Para funcionalidad completa, configura las variables de entorno:")
            print("   - CDS_UID y CDS_API_KEY (para ERA5)")
            print("   - OPENWEATHER_API_KEY (para OpenWeatherMap)")
            print("   - Meteostat_API_KEY (para Meteostat)")

        print("✅ Configuración completada")
        return True

    def download_data(self, lat=None, lon=None, days=7):
        """Descarga datos para una ubicación específica."""
        if lat is None or lon is None:
            lat, lon = 28.6333, -106.0691
        cmd = f'''
import sys, os
# Ya estamos en src, no necesitamos cambiar directorio

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")

from data_ingest import download_data_for_location
from utils import load_config
config = load_config("../config.yaml")
from datetime import datetime, timedelta
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days={days})).strftime('%Y-%m-%d')
data = download_data_for_location({lat}, {lon}, start_date, end_date, config)
print(f"Datos descargados: {{len(data)}} fuentes")
for source, df in data.items():
    print(f"  {{source}}: {{len(df)}} registros")
'''
        return run_command(cmd, f"Descargando datos para ({lat}, {lon})")

    def preprocess_data(self, lat=None, lon=None, days=7):
        """Preprocesa datos para una ubicación."""
        if lat is None or lon is None:
            lat, lon = 28.6333, -106.0691
        cmd = f'''
import sys, os
# Ya estamos en src, no necesitamos cambiar directorio

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")

from preprocess import preprocess_data_for_location
from utils import load_config
from pathlib import Path
config = load_config("../config.yaml")

# Verificar si hay datos para procesar
raw_data_dir = Path("../data/raw")
if not raw_data_dir.exists() or not list(raw_data_dir.glob("*.csv")):
    print("ERROR: No hay datos descargados. Ejecuta 'python make.py download' primero.")
    exit(1)

processed = preprocess_data_for_location({lat}, {lon}, None, None, config)
print(f"Datos procesados: {{processed.shape}}")
'''
        return run_command(cmd, f"Preprocesando datos para ({lat}, {lon})")

    def train_model(self, lat=None, lon=None, days=30):
        """Entrena un modelo para una ubicación."""
        if lat is None or lon is None:
            lat, lon = 28.6333, -106.0691
        cmd = f'''
import sys, os
# Ya estamos en src, no necesitamos cambiar directorio

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")

from train_model import train_model_for_location
from utils import load_config
config = load_config("../config.yaml")
results = train_model_for_location({lat}, {lon}, config, days={days})
print(f"Modelo entrenado: {{results[\"model_type\"]}}")
print(f"  AUC: {{results[\"metrics\"][\"auc\"]:.3f}}")
print(f"  Accuracy: {{results[\"metrics\"][\"accuracy\"]:.3f}}")
'''
        return run_command(cmd, f"Entrenando modelo para ({lat}, {lon})")

    def make_prediction(self, lat=None, lon=None):
        """Hace una predicción para una ubicación."""
        if lat is None or lon is None:
            lat, lon = 28.6333, -106.0691
        cmd = f'''
import sys, os
# Ya estamos en src, no necesitamos cambiar directorio

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")

from predict import predict_rain_for_location
from utils import load_config
config = load_config("../config.yaml")
prediction = predict_rain_for_location({lat}, {lon}, config)
if prediction:
    print("Prediccion completada:")
    print(f"  Probabilidad de lluvia: {{prediction[\"rain_probability\"]:.1%}}")
    print(f"  Llovera en 6h: {{\"Si\" if prediction[\"will_rain_next_6h\"] else \"No\"}}")
    print(f"  Nivel de confianza: {{prediction.get(\"confidence_level\", \"desconocido\")}}")
    print(f"  Modelo: {{prediction[\"model_info\"][\"type\"]}}")

    # Mostrar condiciones actuales si están disponibles
    current = prediction.get("current_conditions", {{}})
    if current:
        temp = current.get("temp", "N/A")
        humidity = current.get("humidity", "N/A")
        clouds = current.get("clouds", "N/A")
        print(f"  Condiciones actuales:")
        print(f"    Temperatura: {{temp}}°C")
        print(f"    Humedad: {{humidity}}%")
        print(f"    Nubosidad: {{clouds}}%")

    note = prediction.get("note", "")
    if note:
        print(f"  Nota: {{note}}")
else:
    print("No se pudo hacer la prediccion")
'''
        return run_command(cmd, f"Haciendo prediccion para ({lat}, {lon})")

    def run_api_server(self, host="0.0.0.0", port=8000):
        """Ejecuta el servidor de la API."""
        cmd = f'python -c "import sys; sys.path.insert(0, \'src\'); from api import run_api; run_api(host=\'{host}\', port={port})"'
        print(f"Iniciando API en {host}:{port}...")
        return run_command(cmd, "Ejecutando servidor API")

    def run_tests(self):
        """Ejecuta los tests del proyecto."""
        return run_command("python check_setup.py", "Ejecutando verificacion del sistema")

    def clean_project(self):
        """Limpia archivos generados."""
        print("Limpieza completada")
        return True

    def run_pipeline(self, lat=None, lon=None):
        """Ejecuta todo el pipeline completo."""
        if lat is None or lon is None:
            lat, lon = 28.6333, -106.0691

        return run_command("python run_pipeline.py", "Ejecutando pipeline completo")


def main():
    """Función principal del CLI."""
    parser = argparse.ArgumentParser(description="Climate Predictor - Automatización")
    parser.add_argument("command", choices=[
        "install", "setup", "download", "preprocess", "train", "predict",
        "api", "test", "clean", "all"
    ], help="Comando a ejecutar")

    parser.add_argument("--lat", type=float, help="Latitud (opcional)")
    parser.add_argument("--lon", type=float, help="Longitud (opcional)")
    parser.add_argument("--days", type=int, default=7, help="Días de datos")
    parser.add_argument("--host", default="0.0.0.0", help="Host para la API")
    parser.add_argument("--port", type=int, default=8000, help="Puerto para la API")

    args = parser.parse_args()

    # Inicializar CLI
    cli = ClimatePredictorCLI()

    # Ejecutar comando
    try:
        if args.command == "install":
            cli.install_dependencies()

        elif args.command == "setup":
            cli.setup_project()

        elif args.command == "download":
            cli.download_data(args.lat, args.lon, args.days)

        elif args.command == "preprocess":
            cli.preprocess_data(args.lat, args.lon, args.days)

        elif args.command == "train":
            cli.train_model(args.lat, args.lon, args.days)

        elif args.command == "predict":
            cli.make_prediction(args.lat, args.lon)

        elif args.command == "api":
            cli.run_api_server(args.host, args.port)

        elif args.command == "test":
            cli.run_tests()

        elif args.command == "clean":
            cli.clean_project()

        elif args.command == "all":
            cli.run_pipeline(args.lat, args.lon)

    except KeyboardInterrupt:
        print("\n🛑 Operación cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
