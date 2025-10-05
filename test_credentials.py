#!/usr/bin/env python3
"""
Script para verificar que las credenciales estén configuradas correctamente
"""

from utils_credentials import credentials_manager

def main():
    print('VERIFICACION DE CREDENCIALES')
    print('=' * 40)

    # Verificar estado de credenciales
    status = credentials_manager.validate_credentials()

    print(f'Copernicus CDS: {"OK" if status["cds"] else "FALTA"}')
    print(f'NASA Earthdata: {"OK" if status["nasa"] else "FALTA"}')
    print()

    if all(status.values()):
        print('EXITO: Todas las credenciales están configuradas correctamente!')
        print('Ahora puedes usar el sistema de predicción meteorológica.')
        print()
        print('Proximos pasos:')
        print('1. python main.py setup    # Configurar base de datos postal')
        print('2. python main.py train    # Entrenar modelo')
        print('3. python main.py predict 31125  # Generar pronóstico')
    else:
        print('ADVERTENCIA: Faltan algunas credenciales. Revisa el archivo .env')
        print()
        print('Contenido esperado en .env:')
        print('# Copernicus CDS API (ERA5)')
        print('CDS_API_KEY=tu_api_key_de_cds')
        print()
        print('# NASA Earthdata (IMERG)')
        print('NASA_EARTHDATA_USERNAME=tu_usuario')
        print('NASA_EARTHDATA_PASSWORD=tu_contraseña')

if __name__ == "__main__":
    main()
