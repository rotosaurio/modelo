"""
Módulo de visualización para pronósticos meteorológicos
Genera gráficos y visualizaciones de los pronósticos
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

from config import PLOT_CONFIG, LOGS_DIR, TIME_STEP_MINUTES, OUTPUT_STEPS

logger = logging.getLogger(__name__)


class WeatherVisualizer:
    """Clase para crear visualizaciones de pronósticos meteorológicos"""

    def __init__(self):
        # Configurar estilo de matplotlib
        plt.style.use(PLOT_CONFIG['style'])
        sns.set_palette("husl")

        # Crear directorio para guardar gráficos
        self.plots_dir = LOGS_DIR / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def create_forecast_plot(self, forecast_result: Dict[str, Any],
                           save_path: Optional[Path] = None) -> Optional[Path]:
        """
        Crea un gráfico completo del pronóstico meteorológico
        Args:
            forecast_result: Resultado de get_weather_forecast()
            save_path: Path donde guardar el gráfico (opcional)
        Returns:
            Path del archivo guardado o None si falla
        """
        try:
            if "error" in forecast_result or "forecast" not in forecast_result:
                logger.error("Resultado de pronóstico inválido para graficar")
                return None

            forecast_data = forecast_result['forecast']
            postal_code = forecast_result['postal_code']

            # Crear DataFrame con los datos del pronóstico
            forecast_df = self._forecast_to_dataframe(forecast_data)

            # Crear figura con subplots
            fig, axes = plt.subplots(3, 2, figsize=PLOT_CONFIG['figsize'], dpi=PLOT_CONFIG['dpi'])
            fig.suptitle(f'Pronóstico Meteorológico - Código Postal {postal_code}',
                        fontsize=16, fontweight='bold', y=0.95)

            # 1. Temperatura
            self._plot_temperature(forecast_df, axes[0, 0])

            # 2. Precipitación
            self._plot_precipitation(forecast_df, axes[0, 1])

            # 3. Humedad
            self._plot_humidity(forecast_df, axes[1, 0])

            # 4. Viento
            self._plot_wind(forecast_df, axes[1, 1])

            # 5. Presión
            self._plot_pressure(forecast_df, axes[2, 0])

            # 6. Nubosidad
            self._plot_cloud_cover(forecast_df, axes[2, 1])

            # Ajustar layout
            plt.tight_layout()

            # Guardar gráfico
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = self.plots_dir / f"forecast_{postal_code}_{timestamp}.png"

            plt.savefig(save_path, bbox_inches='tight', dpi=PLOT_CONFIG['dpi'])
            logger.info(f"Gráfico de pronóstico guardado: {save_path}")

            plt.close()
            return save_path

        except Exception as e:
            logger.error(f"Error creando gráfico de pronóstico: {e}")
            return None

    def _forecast_to_dataframe(self, forecast_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convierte datos de pronóstico a DataFrame"""
        try:
            records = []
            base_time = datetime.utcnow()

            for step_data in forecast_data:
                # Parsear tiempo relativo
                time_str = step_data['time']  # e.g., "+15min"
                minutes_ahead = int(time_str.replace('+', '').replace('min', ''))
                timestamp = base_time + timedelta(minutes=minutes_ahead)

                record = {
                    'timestamp': timestamp,
                    'minutes_ahead': minutes_ahead,
                    'temp_celsius': step_data['temp'],
                    'precipitation_mm': step_data['precipitation_mm'],
                    'humidity_percent': step_data['humidity'],
                    'wind_speed_ms': step_data['wind_speed'],
                    'pressure_hpa': step_data['pressure'],
                    'cloud_cover_percent': step_data['cloud_cover'],
                    'description': step_data['desc']
                }
                records.append(record)

            return pd.DataFrame(records)

        except Exception as e:
            logger.error(f"Error convirtiendo forecast a DataFrame: {e}")
            return pd.DataFrame()

    def _plot_temperature(self, df: pd.DataFrame, ax: plt.Axes):
        """Grafica temperatura"""
        ax.plot(df['timestamp'], df['temp_celsius'],
               color=PLOT_CONFIG['colors']['temperature'], linewidth=2.5, marker='o', markersize=4)

        ax.set_title('Temperatura (°C)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Temperatura (°C)', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Formatear eje X
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(axis='x', rotation=45)

        # Líneas de referencia
        ax.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Caluroso (>30°C)')
        ax.axhline(y=10, color='blue', linestyle='--', alpha=0.5, label='Frío (<10°C)')

        # Añadir valores en puntos
        for i, temp in enumerate(df['temp_celsius']):
            ax.annotate(f'{temp:.1f}°', (df['timestamp'][i], temp),
                       xytext=(0, 5), textcoords='offset points',
                       ha='center', fontsize=8, fontweight='bold')

    def _plot_precipitation(self, df: pd.DataFrame, ax: plt.Axes):
        """Grafica precipitación"""
        bars = ax.bar(df['timestamp'], df['precipitation_mm'],
                     color=PLOT_CONFIG['colors']['precipitation'], alpha=0.7, width=0.02)

        ax.set_title('Precipitación (mm)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Precipitación (mm)', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Formatear eje X
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(axis='x', rotation=45)

        # Añadir valores sobre las barras
        for bar, precip in zip(bars, df['precipitation_mm']):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{precip:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Líneas de umbral
        ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='Lluvia ligera')
        ax.axhline(y=2.5, color='red', linestyle='--', alpha=0.7, label='Lluvia moderada')

    def _plot_humidity(self, df: pd.DataFrame, ax: plt.Axes):
        """Grafica humedad"""
        ax.plot(df['timestamp'], df['humidity_percent'],
               color=PLOT_CONFIG['colors']['humidity'], linewidth=2.5, marker='s', markersize=4)

        ax.set_title('Humedad Relativa (%)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Humedad (%)', fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        # Formatear eje X
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(axis='x', rotation=45)

        # Añadir valores en puntos
        for i, hum in enumerate(df['humidity_percent']):
            ax.annotate(f'{hum:.0f}%', (df['timestamp'][i], hum),
                       xytext=(0, 5), textcoords='offset points',
                       ha='center', fontsize=8)

    def _plot_wind(self, df: pd.DataFrame, ax: plt.Axes):
        """Grafica velocidad del viento"""
        ax.plot(df['timestamp'], df['wind_speed_ms'],
               color=PLOT_CONFIG['colors']['wind'], linewidth=2.5, marker='^', markersize=4)

        ax.set_title('Velocidad del Viento (m/s)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Velocidad (m/s)', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Formatear eje X
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(axis='x', rotation=45)

        # Línea de umbral de viento fuerte
        ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Ventoso (>10 m/s)')

        # Añadir valores en puntos
        for i, wind in enumerate(df['wind_speed_ms']):
            ax.annotate(f'{wind:.1f}', (df['timestamp'][i], wind),
                       xytext=(0, 5), textcoords='offset points',
                       ha='center', fontsize=8)

    def _plot_pressure(self, df: pd.DataFrame, ax: plt.Axes):
        """Grafica presión atmosférica"""
        ax.plot(df['timestamp'], df['pressure_hpa'],
               color=PLOT_CONFIG['colors']['pressure'], linewidth=2.5, marker='d', markersize=4)

        ax.set_title('Presión Atmosférica (hPa)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Presión (hPa)', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Formatear eje X
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(axis='x', rotation=45)

        # Añadir valores en puntos
        for i, pres in enumerate(df['pressure_hpa']):
            ax.annotate(f'{pres:.0f}', (df['timestamp'][i], pres),
                       xytext=(0, 5), textcoords='offset points',
                       ha='center', fontsize=8)

    def _plot_cloud_cover(self, df: pd.DataFrame, ax: plt.Axes):
        """Grafica cobertura de nubes"""
        ax.fill_between(df['timestamp'], df['cloud_cover_percent'],
                       color=PLOT_CONFIG['colors']['clouds'], alpha=0.6)

        ax.plot(df['timestamp'], df['cloud_cover_percent'],
               color=PLOT_CONFIG['colors']['clouds'], linewidth=2, marker='*', markersize=4)

        ax.set_title('Cobertura de Nubes (%)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Nubosidad (%)', fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        # Formatear eje X
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(axis='x', rotation=45)

        # Línea de umbral de nublado
        ax.axhline(y=60, color='gray', linestyle='--', alpha=0.7, label='Nublado (>60%)')

        # Añadir valores en puntos
        for i, cloud in enumerate(df['cloud_cover_percent']):
            ax.annotate(f'{cloud:.0f}%', (df['timestamp'][i], cloud),
                       xytext=(0, 5), textcoords='offset points',
                       ha='center', fontsize=8)

    def create_summary_table(self, forecast_result: Dict[str, Any]) -> str:
        """
        Crea una tabla resumen del pronóstico en formato texto
        Args:
            forecast_result: Resultado de get_weather_forecast()
        Returns:
            String con tabla formateada
        """
        try:
            if "error" in forecast_result:
                return f"Error: {forecast_result['error']}"

            forecast_data = forecast_result['forecast']
            postal_code = forecast_result['postal_code']

            # Crear tabla
            table_lines = []
            table_lines.append("PRONOSTICO METEOROLOGICO")
            table_lines.append("="*80)
            table_lines.append(f"Código Postal: {postal_code}")
            table_lines.append(f"Coordenadas: {forecast_result['coords'][0]}, {forecast_result['coords'][1]}")
            table_lines.append(f"Generado: {forecast_result['generated_at'][:19]} UTC")
            table_lines.append("="*80)

            # Encabezados
            table_lines.append("<15")
            table_lines.append("-"*80)

            # Filas de datos
            for step in forecast_data:
                time = step['time']
                temp = step['temp']
                rain = step['precipitation_mm']
                hum = step['humidity']
                wind = step['wind_speed']
                pres = step['pressure']
                clouds = step['cloud_cover']
                desc = step['desc']

                # Formato visual para lluvia
                rain_icon = "LLUVIA" if rain > 0.1 else "SOLEADO"
                rain_str = f"{rain:.1f}mm"

                table_lines.append(f"{time:>10} | {temp:>5.1f}C | {rain_str:>7} {rain_icon:>8} | {hum:>4.0f}% | {wind:>4.1f}m/s | {pres:>6.0f}hPa | {clouds:>4.0f}% | {desc}")

            table_lines.append("-"*80)
            table_lines.append("Cada fila representa 15 minutos en el futuro")
            table_lines.append("Temperatura en C, precipitacion en mm, humedad/velocidad/nubosidad en %")

            return "\n".join(table_lines)

        except Exception as e:
            logger.error(f"Error creando tabla resumen: {e}")
            return f"Error creando tabla: {e}"

    def create_weather_timeline(self, forecast_result: Dict[str, Any],
                              save_path: Optional[Path] = None) -> Optional[Path]:
        """
        Crea una línea temporal visual del clima con íconos y descripciones
        Args:
            forecast_result: Resultado de get_weather_forecast()
            save_path: Path donde guardar (opcional)
        Returns:
            Path del archivo guardado
        """
        try:
            if "error" in forecast_result or "forecast" not in forecast_result:
                return None

            forecast_data = forecast_result['forecast']
            postal_code = forecast_result['postal_code']

            # Configurar figura
            fig, ax = plt.subplots(figsize=(15, 8), dpi=PLOT_CONFIG['dpi'])

            # Crear timeline
            times = [step['time'] for step in forecast_data]
            temps = [step['temp'] for step in forecast_data]
            rains = [step['precipitation_mm'] for step in forecast_data]

            # Plot de temperatura como línea principal
            ax.plot(range(len(times)), temps, 'b-', linewidth=3, alpha=0.7, label='Temperatura (°C)')

            # Añadir barras de precipitación
            ax2 = ax.twinx()
            bars = ax2.bar(range(len(times)), rains, alpha=0.3, color='blue', width=0.6, label='Precipitación (mm)')

            # Configurar ejes
            ax.set_xlabel('Tiempo', fontsize=12, fontweight='bold')
            ax.set_ylabel('Temperatura (°C)', color='blue', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Precipitación (mm)', color='blue', fontsize=12, fontweight='bold')

            # Configurar ticks del eje X
            ax.set_xticks(range(len(times)))
            ax.set_xticklabels(times, rotation=45, ha='right')

            # Título
            ax.set_title(f'Línea Temporal del Clima - Código Postal {postal_code}',
                        fontsize=16, fontweight='bold', pad=20)

            # Añadir descripciones del clima
            for i, step in enumerate(forecast_data):
                desc = step['desc']
                temp = step['temp']

                # Color basado en temperatura
                if temp > 30:
                    color = 'red'
                elif temp < 10:
                    color = 'blue'
                else:
                    color = 'green'

                # Añadir texto con descripción
                ax.annotate(desc, (i, temp), xytext=(0, 20), textcoords='offset points',
                           ha='center', fontsize=9, fontweight='bold', color=color,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

                # Añadir valor de temperatura
                ax.annotate(f'{temp:.1f}°C', (i, temp), xytext=(0, -15), textcoords='offset points',
                           ha='center', fontsize=8, color='blue')

            # Leyenda
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            # Grid
            ax.grid(True, alpha=0.3)

            # Ajustar layout
            plt.tight_layout()

            # Guardar
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = self.plots_dir / f"timeline_{postal_code}_{timestamp}.png"

            plt.savefig(save_path, bbox_inches='tight', dpi=PLOT_CONFIG['dpi'])
            logger.info(f"Línea temporal guardada: {save_path}")

            plt.close()
            return save_path

        except Exception as e:
            logger.error(f"Error creando línea temporal: {e}")
            return None

    def display_forecast_summary(self, forecast_result: Dict[str, Any]):
        """
        Muestra un resumen completo del pronóstico con tabla y gráfico
        Args:
            forecast_result: Resultado de get_weather_forecast()
        """
        try:
            if "error" in forecast_result:
                print(f"ERROR: {forecast_result['error']}")
                return

            postal_code = forecast_result['postal_code']

            # Mostrar tabla resumen
            print("\n" + self.create_summary_table(forecast_result))

            # Crear y mostrar gráficos
            print("Generando visualizaciones...")
            print("-" * 50)

            # Gráfico completo
            plot_path = self.create_forecast_plot(forecast_result)
            if plot_path:
                print(f"Grafico completo guardado: {plot_path}")

            # Línea temporal
            timeline_path = self.create_weather_timeline(forecast_result)
            if timeline_path:
                print(f"Linea temporal guardada: {timeline_path}")

            print(f"\nPronostico completado para codigo postal {postal_code}")

        except Exception as e:
            logger.error(f"Error mostrando resumen: {e}")
            print(f"Error mostrando resumen: {e}")


def plot_forecast(forecast_result: Dict[str, Any]) -> Optional[Path]:
    """
    Función auxiliar para crear gráfico de pronóstico
    Args:
        forecast_result: Resultado de get_weather_forecast()
    Returns:
        Path del gráfico guardado
    """
    visualizer = WeatherVisualizer()
    return visualizer.create_forecast_plot(forecast_result)


def display_forecast(forecast_result: Dict[str, Any]):
    """
    Función auxiliar para mostrar pronóstico completo
    Args:
        forecast_result: Resultado de get_weather_forecast()
    """
    visualizer = WeatherVisualizer()
    visualizer.display_forecast_summary(forecast_result)
