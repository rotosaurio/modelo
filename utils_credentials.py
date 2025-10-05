"""
Módulo para gestión unificada de credenciales desde archivo .env
Centraliza la lógica de autenticación para todas las APIs
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import dotenv

from config import CDS_API_CONFIG, NASA_CONFIG

logger = logging.getLogger(__name__)


class CredentialsManager:
    """Gestor unificado de credenciales para todas las APIs"""

    def __init__(self, env_file: str = ".env"):
        self.env_file = Path(env_file)
        self._load_env()

    def _load_env(self):
        """Carga variables de entorno desde archivo .env"""
        try:
            if self.env_file.exists():
                dotenv.load_dotenv(self.env_file)
                logger.info(f"Variables de entorno cargadas desde {self.env_file}")
            else:
                logger.warning(f"Archivo .env no encontrado: {self.env_file}")
                logger.info("Creando archivo .env con variables de ejemplo...")

                # Crear archivo .env con variables de ejemplo
                self._create_example_env()

        except Exception as e:
            logger.error(f"Error cargando .env: {e}")

    def _create_example_env(self):
        """Crea archivo .env con variables de ejemplo"""
        try:
            example_content = """# Credenciales para APIs de datos climáticos
# Copernicus CDS API (ERA5)
CDS_API_KEY=tu_uid:tu_api_key

# NASA Earthdata (IMERG)
NASA_EARTHDATA_USERNAME=tu_usuario
NASA_EARTHDATA_PASSWORD=tu_contraseña
"""

            with open(self.env_file, 'w') as f:
                f.write(example_content)

            logger.info(f"Archivo .env creado: {self.env_file}")
            logger.info("Edita este archivo con tus credenciales reales")

        except Exception as e:
            logger.error(f"Error creando archivo .env: {e}")

    def get_cds_credentials(self) -> Optional[str]:
        """
        Obtiene credenciales para Copernicus CDS API
        Returns:
            API key en formato "uid:key" o None si no está disponible
        """
        try:
            api_key = os.getenv(CDS_API_CONFIG["env_key"])
            if api_key:
                logger.info("Credenciales CDS encontradas en variables de entorno")
                return api_key
            else:
                logger.warning(f"Variable de entorno {CDS_API_CONFIG['env_key']} no encontrada")
                return None

        except Exception as e:
            logger.error(f"Error obteniendo credenciales CDS: {e}")
            return None

    def get_nasa_credentials(self) -> Optional[Dict[str, str]]:
        """
        Obtiene credenciales para NASA Earthdata
        Returns:
            Diccionario con bearer_token o None si no está disponible
        """
        try:
            bearer_token = os.getenv(NASA_CONFIG["env_bearer_token"])

            if bearer_token:
                logger.info("Bearer token NASA encontrado en variables de entorno")
                return {
                    "bearer_token": bearer_token
                }
            else:
                logger.warning(f"Variable de entorno {NASA_CONFIG['env_bearer_token']} no encontrada")
                return None

        except Exception as e:
            logger.error(f"Error obteniendo credenciales NASA: {e}")
            return None

    def validate_credentials(self) -> Dict[str, bool]:
        """
        Valida que todas las credenciales estén disponibles
        Returns:
            Diccionario con estado de cada API
        """
        status = {}

        # Validar CDS
        cds_creds = self.get_cds_credentials()
        status["cds"] = cds_creds is not None and len(cds_creds.strip()) > 20

        # Validar NASA
        nasa_creds = self.get_nasa_credentials()
        status["nasa"] = nasa_creds is not None and "bearer_token" in nasa_creds

        return status

    def setup_cds_client(self) -> Optional[Any]:
        """
        Configura cliente CDS con credenciales del .env
        Returns:
            Cliente CDS configurado o None si falla
        """
        try:
            import cdsapi

            api_key = self.get_cds_credentials()
            if not api_key:
                return None

            # Crear cliente con credenciales directas
            client = cdsapi.Client(key=api_key, url=CDS_API_CONFIG["url"])
            logger.info("Cliente CDS configurado correctamente con credenciales de .env")
            return client

        except ImportError:
            logger.error("cdsapi no está instalado")
            return None
        except Exception as e:
            logger.error(f"Error configurando cliente CDS: {e}")
            return None

    def setup_nasa_session(self) -> Optional[Any]:
        """
        Configura sesión NASA con bearer token del .env
        Returns:
            Sesión requests configurada o None si falla
        """
        try:
            import requests

            creds = self.get_nasa_credentials()
            if not creds:
                return None

            # Crear sesión con bearer token
            session = requests.Session()
            session.headers.update({
                "Authorization": f"Bearer {creds['bearer_token']}"
            })

            logger.info("Sesión NASA configurada correctamente con bearer token")
            return session

        except ImportError:
            logger.error("requests no está instalado")
            return None
        except Exception as e:
            logger.error(f"Error configurando sesión NASA: {e}")
            return None


# Instancia global del gestor de credenciales
credentials_manager = CredentialsManager()
