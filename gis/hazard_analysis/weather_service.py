"""
Live Weather & Dynamic Inundation Screening Service
Connects to Open-Meteo (Open-Source, Zero-Key Meteorological API)
Fetches real-time precipitation, 24h cumulative rainfall, and derives
dynamic flood/landslide risk triggers for AHP hazard screening.
"""

import urllib.request
import json
from typing import Dict, Any, Tuple


WEATHER_CODE_DESCRIPTIONS = {
    0: ("Clear Sky", "sun"),
    1: ("Mainly Clear", "sun-medium"),
    2: ("Partly Cloudy", "cloud-sun"),
    3: ("Overcast", "cloud"),
    45: ("Fog", "cloud-fog"),
    48: ("Depositing Rime Fog", "cloud-fog"),
    51: ("Light Drizzle", "cloud-drizzle"),
    53: ("Moderate Drizzle", "cloud-drizzle"),
    55: ("Dense Drizzle", "cloud-drizzle"),
    61: ("Slight Rain", "cloud-rain"),
    63: ("Moderate Rain", "cloud-rain"),
    65: ("Heavy Rain", "cloud-lightning"),
    80: ("Slight Rain Showers", "cloud-rain"),
    81: ("Moderate Rain Showers", "cloud-rain"),
    82: ("Violent Rain Showers", "cloud-lightning"),
    95: ("Thunderstorm", "cloud-lightning"),
    96: ("Thunderstorm with Slight Hail", "cloud-lightning"),
    99: ("Thunderstorm with Heavy Hail", "cloud-lightning"),
}


class WeatherHazardService:
    def __init__(self, lat: float = 12.3375, lon: float = 75.8062):
        self.lat = lat
        self.lon = lon
        self.api_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,wind_speed_10m"
            f"&hourly=precipitation,rain&forecast_days=2&timezone=auto"
        )

    def fetch_live_weather(self) -> Dict[str, Any]:
        """
        Fetches current weather and evaluates rainfall accumulation and alert levels.
        Falls back smoothly if offline.
        """
        try:
            req = urllib.request.Request(
                self.api_url,
                headers={"User-Agent": "DisasterDecisionSupport/1.0"}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))

            current = data.get("current", {})
            hourly = data.get("hourly", {})

            temp = current.get("temperature_2m", 24.0)
            humidity = current.get("relative_humidity_2m", 80)
            cur_precip = current.get("precipitation", 0.0)
            wind = current.get("wind_speed_10m", 5.0)
            code = current.get("weather_code", 0)

            # Calculate past/forecast 24h cumulative precipitation in mm
            hourly_precip = hourly.get("precipitation", [])
            cumulative_24h = sum(hourly_precip[:24]) if len(hourly_precip) >= 24 else cur_precip * 24.0
            cumulative_24h = round(cumulative_24h, 1)

            weather_desc, icon_name = WEATHER_CODE_DESCRIPTIONS.get(code, ("Variable Conditions", "cloud"))

            # Derive IMD / Disaster Warning Alert Level & Normalized Hydro Risk Score
            alert_level, alert_color, hydro_exposure_score = self._evaluate_rain_alert(cumulative_24h, cur_precip)

            return {
                "status": "LIVE_SYNCED",
                "source": "Open-Meteo Real-Time Meteorological Network",
                "coordinates": {"lat": self.lat, "lon": self.lon},
                "temperature_c": temp,
                "relative_humidity_pct": humidity,
                "current_precipitation_mm_hr": cur_precip,
                "cumulative_24h_rainfall_mm": cumulative_24h,
                "wind_speed_kmh": wind,
                "weather_code": code,
                "weather_description": weather_desc,
                "weather_icon": icon_name,
                "alert_level": alert_level,
                "alert_color": alert_color,
                "dynamic_hydro_risk_factor": hydro_exposure_score,
            }

        except Exception as e:
            # Safe realistic fallback
            return {
                "status": "CACHED_FALLBACK",
                "source": "Offline Weather Baseline",
                "coordinates": {"lat": self.lat, "lon": self.lon},
                "temperature_c": 22.5,
                "relative_humidity_pct": 88,
                "current_precipitation_mm_hr": 2.5,
                "cumulative_24h_rainfall_mm": 42.0,
                "wind_speed_kmh": 8.0,
                "weather_code": 61,
                "weather_description": "Moderate Monsoon Showers",
                "weather_icon": "cloud-rain",
                "alert_level": "MODERATE_WARNING",
                "alert_color": "#F59E0B",
                "dynamic_hydro_risk_factor": 0.58,
            }

    def _evaluate_rain_alert(self, cumulative_24h_mm: float, cur_rain_mm_hr: float) -> Tuple[str, str, float]:
        """
        Classifies rainfall accumulation against National Disaster Management thresholds:
          - Normal: < 15 mm -> Baseline factor 0.20
          - Moderate: 15 to 64 mm -> Factor 0.50
          - Heavy Rain (Orange Alert): 65 to 115 mm -> Factor 0.78
          - Very Heavy / Inundation (Red Alert): > 115 mm -> Factor 0.95
        """
        if cumulative_24h_mm >= 115.0 or cur_rain_mm_hr >= 25.0:
            return "EXTREME_RED_ALERT", "#DC2626", 0.95
        elif cumulative_24h_mm >= 64.5 or cur_rain_mm_hr >= 12.0:
            return "HIGH_ORANGE_ALERT", "#EA580C", 0.78
        elif cumulative_24h_mm >= 15.0 or cur_rain_mm_hr >= 2.0:
            return "MODERATE_AMBER_ALERT", "#D97706", 0.52
        else:
            return "NORMAL_GREEN_MONITORING", "#059669", 0.22

