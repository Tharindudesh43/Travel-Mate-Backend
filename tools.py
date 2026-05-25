import os
import requests
from typing import Dict, Any, Union, List
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from bs4 import BeautifulSoup

# Load env
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, ".env"))

TAVILY_API_KEY       = os.getenv("TAVILY_API_KEY")
RAPIDAPI_KEY         = os.getenv("RAPIDAPI_KEY")
GROQ_API_KEY         = os.getenv("GROQ_API_KEY")

os.environ["GROQ_API_KEY"]    = GROQ_API_KEY   or ""
os.environ["TAVILY_API_KEY"]  = TAVILY_API_KEY  or ""


# Pydantic Schemas
class GeocodingInput(BaseModel):
    destination: str = Field(description="Sri Lankan place name e.g. 'Kandy, Sri Lanka'")

class WeatherInput(BaseModel):
    latitude:    float = Field(description="Latitude from GetGeocodingOfLocation")
    longitude:   float = Field(description="Longitude from GetGeocodingOfLocation")
    destination: str   = Field(description="Place name e.g. 'Colombo, Sri Lanka'")

class HotelSearchInput(BaseModel):
    latitude:    float = Field(description="Latitude from GetGeocodingOfLocation")
    longitude:   float = Field(description="Longitude from GetGeocodingOfLocation")
    destination: str   = Field(description="Place name e.g. 'Kandy, Sri Lanka'")

class WebSearchInput(BaseModel):
    query: str = Field(description="Travel question about Sri Lanka e.g. 'best beaches in Sri Lanka'")

class EmergencyInput(BaseModel):
    user_query: str = Field(description="The user's emergency message exactly as typed")

class TrainScheduleInput(BaseModel):
    origin:      str = Field(description="Departure city e.g. 'Colombo'. Default to 'Colombo' if not mentioned.")
    destination: str = Field(description="Arrival city e.g. 'Ella', 'Galle', 'Kandy'")

class EmergencyAnalysis(BaseModel):
    emergency_type:     str       = Field(description="Type: police, medical, fire, disaster, embassy, tourist, or general")
    severity:           str       = Field(description="Severity: critical, urgent, or informational")
    primary_contact:    str       = Field(description="Single most important number to call right now")
    secondary_contacts: List[str] = Field(description="2-3 backup contact numbers")
    instructions:       str       = Field(description="3-5 numbered immediate safety steps")
    advice:             str       = Field(description="One calming reassurance sentence")

class BusRouteInput(BaseModel):
    origin:      str = Field(description="Departure city e.g. 'Colombo'. Default to 'Colombo' if not mentioned.")
    destination: str = Field(description="Arrival city e.g. 'Kandy', 'Galle', 'Ella'")
    
    

# WMO WEATHER CODES  (Open-Meteo)
WMO_CODES = {
    0:  {"label": "Clear Sky",             "emoji": "☀️",  "raining": False},
    1:  {"label": "Mainly Clear",          "emoji": "🌤️", "raining": False},
    2:  {"label": "Partly Cloudy",         "emoji": "⛅",  "raining": False},
    3:  {"label": "Overcast",              "emoji": "☁️",  "raining": False},
    45: {"label": "Foggy",                 "emoji": "🌫️", "raining": False},
    48: {"label": "Icy Fog",               "emoji": "🌫️", "raining": False},
    51: {"label": "Light Drizzle",         "emoji": "🌦️", "raining": True},
    53: {"label": "Moderate Drizzle",      "emoji": "🌦️", "raining": True},
    55: {"label": "Heavy Drizzle",         "emoji": "🌧️", "raining": True},
    61: {"label": "Light Rain",            "emoji": "🌧️", "raining": True},
    63: {"label": "Moderate Rain",         "emoji": "🌧️", "raining": True},
    65: {"label": "Heavy Rain",            "emoji": "🌧️", "raining": True},
    80: {"label": "Light Rain Showers",    "emoji": "🌦️", "raining": True},
    81: {"label": "Moderate Rain Showers", "emoji": "🌧️", "raining": True},
    82: {"label": "Heavy Rain Showers",    "emoji": "⛈️",  "raining": True},
    95: {"label": "Thunderstorm",          "emoji": "⛈️",  "raining": True},
    96: {"label": "Thunderstorm + Hail",   "emoji": "⛈️",  "raining": True},
    99: {"label": "Thunderstorm + Hail",   "emoji": "⛈️",  "raining": True},
}



# STATION DATA  (Sri Lanka Railways)
STATION_IDS = {
    "COLOMBO FORT": 61,  "KANDY": 105,      "ELLA": 80,
    "GALLE": 88,         "MATARA": 187,     "NANUOYA": 155,
    "BADULLA": 28,       "HAPUTALE": 97,    "BANDARAWELA": 33,
    "HATTON": 99,        "TRINCOMALEE": 243,"BATTICALOA": 36,
    "JAFFNA": 103,       "ANURADHAPURA": 18,"POLONNARUWA": 193,
    "NEGOMBO": 152,      "HIKKADUWA": 100,  "BENTOTA": 42,
    "MIRISSA": 144,      "UNAWATUNA": 247,  "VAVUNIYA": 250,
    "KURUNEGALA": 128,   "GAMPAHA": 89,     "MARADANA": 136,
    "DEHIWALA": 65,      "MOUNT LAVINIA": 147, "MORATUWA": 145,
    "PANADURA": 175,     "ALUTHGAMA": 12,   "AMBALANGODA": 14,
    "KATUNAYAKE": 112,   "RAGAMA": 199,     "POLGAHAWELA": 192,
    "RAMBUKKANA": 201,   "PERADENIYA": 182, "NAWALAPITIYA": 151,
    "TALAWAKELE": 233,   "GAMPOLA": 91,     "KADUGANNAWA": 104,
    "HABARANA": 96,      "MATALE": 137,     "KOLLUPITIYA": 122,
    "BAMBALAPITIYA": 31, "WELLAWATTA": 258, "WADDUWA": 252,
    "CHILAW": 57,        "PUTTALAM": 197,   "MEDAWACHCHIYA": 139,
    "MAHO": 133,         "AVISSAWELLA": 27,
}


STATION_ALIASES = {
    "COLOMBO":         "COLOMBO FORT",
    "NUWARA ELIYA":    "NANUOYA",
    "SIGIRIYA":        "HABARANA",
    "YALA":            "MATARA",
    "MIRISSA BEACH":   "MIRISSA",
    "UNAWATUNA BEACH": "UNAWATUNA",
    "AIRPORT":         "KATUNAYAKE",
    "CMB":             "KATUNAYAKE",
    "HILL COUNTRY":    "NANUOYA",
    "TEA COUNTRY":     "HATTON",
    "FORT":            "COLOMBO FORT",
}



BUS_ROUTES = {
    # Expressway Routes (Highway) 
    ("COLOMBO", "GALLE"): {
        "route_number": "E01",
        "route_name":   "Colombo → Galle Expressway",
        "bus_stand":    "Colombo: Bastian Mawatha | Galle: Galle Bus Stand",
        "duration":     "1.5 hrs",
        "distance":     "116 km",
        "fare": {
            "SLTB AC":    "Rs. 300",
            "Private AC": "Rs. 350–400",
        },
        "frequency":    "Every 30 mins",
        "first_bus":    "5:30 AM",
        "last_bus":     "10:00 PM",
        "type":         "Expressway",
        "tips":         "Take Southern Expressway — much faster than coastal road.",
    },
    ("COLOMBO", "KANDY"): {
        "route_number": "1",
        "route_name":   "Colombo → Kandy",
        "bus_stand":    "Colombo: Pettah Bus Stand | Kandy: Goods Shed Bus Stand",
        "duration":     "3–3.5 hrs",
        "distance":     "116 km",
        "fare": {
            "SLTB Normal": "Rs. 200",
            "SLTB AC":     "Rs. 350",
            "Private AC":  "Rs. 400–500",
        },
        "frequency":    "Every 15 mins",
        "first_bus":    "5:00 AM",
        "last_bus":     "9:00 PM",
        "type":         "Normal + AC",
        "tips":         "Avoid peak hours 7–9 AM. AC buses are more comfortable.",
    },
    ("COLOMBO", "MATARA"): {
        "route_number": "E02",
        "route_name":   "Colombo → Matara Expressway",
        "bus_stand":    "Colombo: Bastian Mawatha | Matara: Matara Bus Stand",
        "duration":     "2.5 hrs",
        "distance":     "160 km",
        "fare": {
            "SLTB AC":    "Rs. 400",
            "Private AC": "Rs. 450–550",
        },
        "frequency":    "Every 45 mins",
        "first_bus":    "5:00 AM",
        "last_bus":     "9:00 PM",
        "type":         "Expressway",
        "tips":         "Fastest route to south. Book seat in advance for weekends.",
    },
    ("COLOMBO", "JAFFNA"): {
        "route_number": "15",
        "route_name":   "Colombo → Jaffna",
        "bus_stand":    "Colombo: Pettah | Jaffna: Jaffna Bus Stand",
        "duration":     "6–7 hrs",
        "distance":     "398 km",
        "fare": {
            "SLTB Normal": "Rs. 600",
            "SLTB AC":     "Rs. 1000",
            "Private AC":  "Rs. 1200–1500",
        },
        "frequency":    "Every 2 hrs",
        "first_bus":    "6:00 AM",
        "last_bus":     "11:00 PM",
        "type":         "Long Distance + Night Service",
        "tips":         "Night buses available. Book in advance at sltb.eseat.lk",
    },
    ("COLOMBO", "TRINCOMALEE"): {
        "route_number": "48",
        "route_name":   "Colombo → Trincomalee",
        "bus_stand":    "Colombo: Pettah | Trincomalee: Trinco Bus Stand",
        "duration":     "5–6 hrs",
        "distance":     "257 km",
        "fare": {
            "SLTB Normal": "Rs. 500",
            "SLTB AC":     "Rs. 800",
        },
        "frequency":    "Every 2 hrs",
        "first_bus":    "5:30 AM",
        "last_bus":     "9:00 PM",
        "type":         "Long Distance",
        "tips":         "Scenic route through Habarana. Book AC bus for comfort.",
    },
    ("COLOMBO", "ANURADHAPURA"): {
        "route_number": "4",
        "route_name":   "Colombo → Anuradhapura",
        "bus_stand":    "Colombo: Pettah | Anuradhapura: New Bus Stand",
        "duration":     "4–5 hrs",
        "distance":     "205 km",
        "fare": {
            "SLTB Normal": "Rs. 380",
            "SLTB AC":     "Rs. 600",
        },
        "frequency":    "Every 1 hr",
        "first_bus":    "5:00 AM",
        "last_bus":     "8:00 PM",
        "type":         "Long Distance",
        "tips":         "Direct buses available. Ask for 'New Bus Stand' stop.",
    },
    ("COLOMBO", "NUWARA ELIYA"): {
        "route_number": "99",
        "route_name":   "Colombo → Nuwara Eliya",
        "bus_stand":    "Colombo: Pettah | Nuwara Eliya: Nuwara Eliya Bus Stand",
        "duration":     "4–5 hrs",
        "distance":     "180 km",
        "fare": {
            "SLTB Normal": "Rs. 350",
            "SLTB AC":     "Rs. 550",
        },
        "frequency":    "Every 1–2 hrs",
        "first_bus":    "6:00 AM",
        "last_bus":     "7:00 PM",
        "type":         "Hill Country",
        "tips":         "Scenic mountain roads. Can be slow in bad weather.",
    },
    ("KANDY", "ELLA"): {
        "route_number": "N/A",
        "route_name":   "Kandy → Ella",
        "bus_stand":    "Kandy: Goods Shed | Ella: Ella Bus Stop",
        "duration":     "4–5 hrs",
        "distance":     "140 km",
        "fare": {
            "SLTB Normal": "Rs. 280",
        },
        "frequency":    "Every 2 hrs",
        "first_bus":    "7:00 AM",
        "last_bus":     "4:00 PM",
        "type":         "Hill Country",
        "tips":         "Change at Badulla for Ella. Train is more scenic alternative.",
    },
    ("COLOMBO", "NEGOMBO"): {
        "route_number": "240",
        "route_name":   "Colombo → Negombo",
        "bus_stand":    "Colombo: Pettah | Negombo: Negombo Bus Stand",
        "duration":     "1.5–2 hrs",
        "distance":     "37 km",
        "fare": {
            "SLTB Normal": "Rs. 80",
            "Private":     "Rs. 100",
        },
        "frequency":    "Every 15 mins",
        "first_bus":    "5:00 AM",
        "last_bus":     "10:00 PM",
        "type":         "Short Distance",
        "tips":         "Very frequent. No need to book in advance.",
    },
    ("COLOMBO", "HIKKADUWA"): {
        "route_number": "2",
        "route_name":   "Colombo → Hikkaduwa",
        "bus_stand":    "Colombo: Pettah | Hikkaduwa: Main Road Stop",
        "duration":     "2–2.5 hrs",
        "distance":     "100 km",
        "fare": {
            "SLTB Normal": "Rs. 180",
            "SLTB AC":     "Rs. 280",
        },
        "frequency":    "Every 30 mins",
        "first_bus":    "5:30 AM",
        "last_bus":     "9:00 PM",
        "type":         "Coastal",
        "tips":         "Galle road buses stop at Hikkaduwa. Ask driver to confirm.",
    },
}

# Alias map — common names → DB keys
BUS_ALIASES = {
    "FORT":          "COLOMBO",
    "CMB":           "COLOMBO",
    "COLOMBO FORT":  "COLOMBO",
    "NUWARA ELIYA":  "NUWARA ELIYA",
    "NUWARAELIYA":   "NUWARA ELIYA",
    "HIKKADUA":      "HIKKADUWA",
    "GALLE":         "GALLE",
    "MATARA":        "MATARA",
    "MIRISSA":       "MATARA", 
    "UNAWATUNA":     "GALLE",    
    "ELLA":          "ELLA",
    "BADULLA":       "ELLA",
}




# Private helper functions for tools
def _get_geocoding(destination: str) -> Dict[str, Any]:
    """Geocode a Sri Lankan destination using Open-Meteo (free, no API key)."""
    print(f"📍 Geocoding: '{destination}'")
    url    = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": destination, "count": 1, "language": "en", "format": "json"}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("results")
        if results:
            r = results[0]
            print(f"✅ Geocoded: {r['latitude']}, {r['longitude']}")
            return {
                "destination":  destination,
                "latitude":     r["latitude"],
                "longitude":    r["longitude"],
                "display_name": r.get("name", destination),
                "country":      r.get("country", ""),
            }
        return {"error": f"No results for '{destination}'", "destination": destination, "latitude": None, "longitude": None}
    except Exception as e:
        return {"error": str(e), "destination": destination, "latitude": None, "longitude": None}



def _get_weather(latitude: float, longitude: float, destination: str) -> Dict[str, Any]:
    """Fetch current weather using Open-Meteo API (free, no API key)."""
    print(f"🌤️ Weather: {destination} ({latitude}, {longitude})")
    url    = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":   latitude,
        "longitude":  longitude,
        "current":    ["temperature_2m","relative_humidity_2m","apparent_temperature",
                       "precipitation","rain","weather_code","wind_speed_10m","uv_index"],
        "daily":      ["temperature_2m_max","temperature_2m_min",
                       "precipitation_probability_max","weather_code"],
        "timezone":   "Asia/Colombo",
        "forecast_days": 3,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data    = resp.json()
        current = data.get("current", {})
        daily   = data.get("daily",   {})

        code       = current.get("weather_code", 0)
        wmo        = WMO_CODES.get(code, {"label": "Unknown", "emoji": "🌡️", "raining": False})
        temp       = current.get("temperature_2m")
        feels_like = current.get("apparent_temperature")
        humidity   = current.get("relative_humidity_2m")
        rain_mm    = current.get("rain", 0)
        wind       = current.get("wind_speed_10m")
        uv         = current.get("uv_index", 0)
        is_raining = wmo["raining"] or (rain_mm or 0) > 0

        # 3-day forecast
        forecast = []
        for i in range(min(3, len(daily.get("time", [])))):
            dw  = WMO_CODES.get(daily["weather_code"][i] if i < len(daily.get("weather_code",[])) else 0,
                                {"label":"Unknown","emoji":"🌡️"})
            forecast.append({
                "date":       daily["time"][i],
                "condition":  dw["label"],
                "emoji":      dw["emoji"],
                "temp_max":   daily["temperature_2m_max"][i]              if i < len(daily.get("temperature_2m_max",[])) else None,
                "temp_min":   daily["temperature_2m_min"][i]              if i < len(daily.get("temperature_2m_min",[])) else None,
                "rain_chance":daily["precipitation_probability_max"][i]   if i < len(daily.get("precipitation_probability_max",[])) else 0,
            })

        # Travel advice
        tips = []
        if is_raining:                      tips.append("🌂 Bring a raincoat or umbrella")
        if temp and temp > 32:              tips.append("🥵 Very hot — stay hydrated")
        elif temp and temp < 18:            tips.append("🧥 Cool — bring a light jacket")
        if humidity and humidity > 80:      tips.append("💧 Very humid — breathable clothing recommended")
        if uv and uv >= 8:                  tips.append("🧴 Extreme UV — sunscreen essential")
        if "Thunderstorm" in wmo["label"]:  tips.append("⚡ Thunderstorm — avoid open areas")
        if not tips:                        tips.append("✅ Good conditions for sightseeing")

        rain_answer = "Yes ☔ it is currently raining." if is_raining else "No 🌤️ it is not raining right now."

        summary_lines = [
            f"{wmo['emoji']} **Weather in {destination}**",
            f"🌡️ Temperature: {temp}°C  (feels like {feels_like}°C)",
            f"💧 Humidity:    {humidity}%",
            f"🌧️ Raining:     {rain_answer}",
            f"🌬️ Wind:        {wind} km/h  |  ☀️ UV: {uv}",
            f"📋 Condition:   {wmo['label']}",
            f"💡 {' | '.join(tips)}",
            "", "📅 **3-Day Forecast:**",
        ]
        for d in forecast:
            summary_lines.append(
                f"  {d['emoji']} {d['date']}  {d['temp_min']}°C–{d['temp_max']}°C  🌧️{d['rain_chance']}% ({d['condition']})"
            )

        return {
            "destination": destination,
            "weather":     wmo["label"],
            "temperature": temp,
            "feels_like":  feels_like,
            "humidity":    humidity,
            "rain_mm":     rain_mm,
            "is_raining":  is_raining,
            "wind_speed":  wind,
            "uv_index":    uv,
            "forecast":    forecast,
            "summary":     "\n".join(summary_lines),
        }
    except Exception as e:
        print(f"🔥 Weather error: {e}")
        return {"error": str(e), "destination": destination, "weather": None}



def _search_hotels(latitude: float, longitude: float, destination: str) -> Union[List[Dict], str]:
    """Search hotels via Booking.com RapidAPI."""
    print(f"🏨 Hotels: {destination} ({latitude}, {longitude})")
    today     = datetime.today().date()
    arrival   = today + timedelta(days=1)
    departure = today + timedelta(days=2)
    bbox      = f"{float(latitude)-0.1},{float(latitude)+0.1},{float(longitude)-0.1},{float(longitude)+0.1}"

    try:
        resp = requests.get(
            "https://apidojo-booking-v1.p.rapidapi.com/properties/list-by-map",
            headers={"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "apidojo-booking-v1.p.rapidapi.com"},
            params={
                "arrival_date": arrival, "departure_date": departure,
                "latitude": str(latitude), "longitude": str(longitude),
                "room_number": "1", "adults_number": "1", "bbox": bbox,
                "order_by": "popularity", "language_code": "en-us",
                "travel_purpose": "leisure", "units": "metric", "offset": "0",
            },
            timeout=10,
        )
        resp.raise_for_status()
        hotels = resp.json().get("result", [])[:5]
        if not hotels:
            return [{"error": f"No hotels found near {destination}."}]
        return [{
            "name":  h.get("hotel_name", "N/A"),
            "stars": h.get("class", "N/A"),
            "price": f"{h.get('min_total_price','N/A')} {h.get('currency_code','USD')}",
            "url":   h.get("url", "N/A"),
            "image": h.get("max_1440_photo_url") or h.get("main_photo_url", "N/A"),
        } for h in hotels]
    except Exception as e:
        return [{"error": f"Hotel search failed: {str(e)}"}]


def _web_search(query: str) -> str:
    """Search web for Sri Lanka travel info via Tavily."""
    print(f"🔍 Web search: {query}")
    try:
        search  = TavilySearch(max_results=3, search_depth="advanced")
        results = search.run(query)

        # TavilySearch.run() can return dict or string
        if isinstance(results, dict) and "results" in results:
            lines = [f"### Web Search Results for: {query}\n"]
            for r in results["results"]:
                lines.append(f"- **{r.get('title','')}**: {r.get('content','')}\n  Source: {r.get('url','')}")
            return "\n".join(lines)
        elif isinstance(results, str):
            return results
        return f"No specific travel info found for '{query}'"
    except Exception as e:
        return f"WEB_ERROR:: {str(e)}"


def _emergency_data(user_query: str) -> str:
    """Provide Sri Lanka emergency contacts using LLM analysis."""
    contacts_reference = {
        "police":        {"primary": "119",          "secondary": ["118", "011-2433333"]},
        "tourist_police":{"primary": "011-2421052",  "secondary": ["119"]},
        "medical":       {"primary": "1990",         "secondary": ["110", "011-2691111"]},
        "fire_rescue":   {"primary": "110",          "secondary": ["119"]},
        "accident":      {"primary": "011-2691111",  "secondary": ["1990","110"]},
        "report_crime":  {"primary": "011-2691500",  "secondary": ["119","011-5717171"]},
    }
    try:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0).with_structured_output(EmergencyAnalysis)
        analysis = llm.invoke(f"""
You are an emergency assistant for Sri Lanka travelers.
Message: "{user_query}"
Contacts: {contacts_reference}
Rules:
- Physical danger → primary_contact = 119 or 1990
- Tourist theft/scam → primary_contact = 011-2421052
- Medical → primary_contact = 1990
Give 3-5 numbered safety steps. One calming advice sentence.
""")
        icon = {"critical": "🆘", "urgent": "🚨", "informational": "ℹ️"}.get(analysis.severity.lower(), "🚨")
        return "\n".join([
            f"{icon} **EMERGENCY: {analysis.emergency_type.upper()}**",
            f"📞 **Call Now:** {analysis.primary_contact}",
            f"📋 **Backup:** {' | '.join(analysis.secondary_contacts)}",
            f"🛡️ **Steps:** {analysis.instructions}",
            f"💬 {analysis.advice}",
            "━━━━━━━━━━━━━━━━━━━━━━━",
            "🇱🇰 Police:119 | Ambulance:1990 | Fire:110 | Tourist Police:011-2421052",
        ])
    except Exception as e:
        print(f"❌ EmergencyData error: {e}")
        return (
            "🆘 **EMERGENCY — Call Immediately:**\n"
            "📞 Police: 119  |  Ambulance: 1990 (free)  |  Fire: 110\n"
            "📞 Tourist Police: 011-2421052  |  Disaster: 117\n"
            "Stay calm. Move to a safe location."
        )


def resolve_station(name: str):
    name_upper = name.upper().strip()
    canonical  = STATION_ALIASES.get(name_upper, name_upper)
    station_id = STATION_IDS.get(canonical)
    if not station_id:
        for station, sid in STATION_IDS.items():
            if name_upper in station or station in name_upper:
                canonical, station_id = station, sid
                break
    return canonical, station_id


def _get_train_schedule(origin: str, destination: str) -> str:
    """Scrape Sri Lanka Railways eservices portal for train schedule."""
    print(f"🚂 Train: {origin} → {destination}")
    origin_name, origin_id         = resolve_station(origin)
    destination_name, destination_id = resolve_station(destination)

    if not origin_id:
        return f"❌ Station not found: '{origin}'"
    if not destination_id:
        return f"❌ Station not found: '{destination}'"

    try:
        resp = requests.get(
            "https://eservices.railway.gov.lk/schedule/searchTrain.action",
            params={
                "lang": "en", "selectedLocale": "en",
                "searchCriteria.startStationID": origin_id,
                "searchCriteria.endStationID":   destination_id,
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=40,
        )
        resp.raise_for_status()
        return _format_train_response(_parse_schedule_html(resp.text, origin_name, destination_name))
    except requests.exceptions.Timeout:
        return "❌ Sri Lanka Railways timed out.\n📌 Try: https://eservices.railway.gov.lk"
    except Exception as e:
        return f"❌ Train schedule error: {str(e)}\n📌 Try: https://eservices.railway.gov.lk"



def _parse_schedule_html(html: str, origin: str, destination: str) -> Dict[str, Any]:
    soup   = BeautifulSoup(html, "html.parser")
    trains = []
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for i, row in enumerate(rows):
            cols = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cols) >= 6 and ":" in (cols[1] if len(cols) > 1 else ""):
                detail_row = rows[i + 1] if i + 1 < len(rows) else None
                classes, train_no = [], ""
                if detail_row:
                    dt = detail_row.get_text()
                    if "Available Classes:" in dt:
                        classes = [c.strip() for c in dt.split("Available Classes:")[1].split("Train ends")[0].split(",") if "Class" in c]
                    if "Train No:" in dt:
                        train_no = dt.split("Train No:")[1].strip().split()[0]
                trains.append({
                    "departure_time":    cols[2] if len(cols) > 2 else "",
                    "arrival_time":      cols[1] if len(cols) > 1 else "",
                    "frequency":         cols[5] if len(cols) > 5 else "",
                    "train_name":        cols[6] if len(cols) > 6 else "",
                    "train_type":        cols[7] if len(cols) > 7 else "",
                    "train_number":      train_no,
                    "available_classes": classes,
                })
    return {"origin": origin, "destination": destination,
            "date": datetime.now().strftime("%d/%m/%Y"), "trains": trains, "total_trains": len(trains)}



def _format_train_response(data: Dict[str, Any]) -> str:
    if not data.get("trains"):
        return f"❌ No trains: {data['origin']} → {data['destination']}.\n📌 https://eservices.railway.gov.lk"
    lines = [
        f"🚂 **Trains: {data['origin']} → {data['destination']}**",
        f"📅 {data['date']}  |  🔢 {data['total_trains']} train(s)\n",
    ]
    for i, t in enumerate(data["trains"], 1):
        lines += [
            f"**{i}. {t['train_name'] or 'Train '+t['train_number']}** ({t['train_type']})",
            f"   🕐 Departs: {t['departure_time']}  →  Arrives: {t['arrival_time']}",
            f"   📋 Frequency: {t['frequency']}",
            f"   🎫 Classes: {', '.join(t['available_classes']) or 'N/A'}",
            "",
        ]
    lines.append("📌 Book: https://eservices.railway.gov.lk")
    return "\n".join(lines)




def _get_bus_route(origin: str, destination: str) -> Dict[str, Any]:
    """
    Get Sri Lanka bus route info.
    Tries static DB first, falls back to web search.
    """
    print(f"🚌 Bus route: {origin} → {destination}")

    # Normalize input
    origin_key      = BUS_ALIASES.get(origin.upper().strip(), origin.upper().strip())
    destination_key = BUS_ALIASES.get(destination.upper().strip(), destination.upper().strip())

    # Try direct route
    route = (
        BUS_ROUTES.get((origin_key, destination_key)) or
        BUS_ROUTES.get((destination_key, origin_key))
    )

    if route:
        print(f"✅ Found in static DB")
        return {
            "found":          True,
            "origin":         origin,
            "destination":    destination,
            "route_number":   route["route_number"],
            "route_name":     route["route_name"],
            "bus_stand":      route["bus_stand"],
            "duration":       route["duration"],
            "distance":       route["distance"],
            "fare":           route["fare"],
            "frequency":      route["frequency"],
            "first_bus":      route["first_bus"],
            "last_bus":       route["last_bus"],
            "type":           route["type"],
            "tips":           route["tips"],
            "booking_url":    "https://sltb.eseat.lk",
            "hotline":        "1315",
            "summary":        _format_bus_response(origin, destination, route),
        }

    # Not in static DB — web search fallback
    print(f"⚠️ Not in static DB — trying web search")
    return _bus_web_search_fallback(origin, destination)



def _bus_web_search_fallback(origin: str, destination: str) -> Dict[str, Any]:
    """Web search fallback for routes not in static DB."""
    try:
        from langchain_tavily import TavilySearch
        search  = TavilySearch(max_results=2, search_depth="basic")
        results = search.run(
            f"bus route {origin} to {destination} Sri Lanka fare schedule"
        )
        return {
            "found":       False,
            "origin":      origin,
            "destination": destination,
            "summary": (
                f"🚌 **Bus: {origin} → {destination}**\n\n"
                f"{results}\n\n"
                f"📞 For exact times: Call **1315** (SLTB Hotline)\n"
                f"🌐 Book online: https://sltb.eseat.lk"
            ),
        }
    except Exception as e:
        return {
            "found":   False,
            "origin":  origin,
            "destination": destination,
            "summary": (
                f"🚌 Bus route info for {origin} → {destination} not available.\n"
                f"📞 Call SLTB: **1315**\n"
                f"🌐 Book: https://sltb.eseat.lk"
            ),
        }
        


def _format_bus_response(origin: str, destination: str, route: dict) -> str:
    """Format bus route into readable response."""
    fare_lines = "\n".join(
        f"   • {k}: {v}" for k, v in route["fare"].items()
    )
    return "\n".join([
        f"🚌 **Bus: {origin} → {destination}**",
        f"",
        f"🔢 Route:      {route['route_number']} — {route['route_name']}",
        f"⏱️ Duration:   {route['duration']}",
        f"📏 Distance:   {route['distance']}",
        f"🕐 First Bus:  {route['first_bus']}  |  Last Bus: {route['last_bus']}",
        f"🔄 Frequency:  {route['frequency']}",
        f"🚏 Bus Stands: {route['bus_stand']}",
        f"",
        f"💰 **Fares:**",
        fare_lines,
        f"",
        f"💡 **Tip:** {route['tips']}",
        f"",
        f"📞 Hotline: 1315  |  🌐 Book: https://sltb.eseat.lk",
    ])


# Public Tools (LLM sees these)
GetGeocodingOfLocation = StructuredTool.from_function(
    func=        _get_geocoding,
    name=        "GetGeocodingOfLocation",
    description= "Get GPS coordinates for a Sri Lankan place. ALWAYS call first before hotels or weather. Use for any city, landmark, beach, temple, or park in Sri Lanka.",
    args_schema= GeocodingInput,
)

GetWeatherOfDestination = StructuredTool.from_function(
    func=        _get_weather,
    name=        "GetWeatherOfDestination",
    description= "Get current weather for a Sri Lankan destination. Use for: rain, raining, weather, temperature, hot, cold, humid, sunny, cloudy, forecast, umbrella, monsoon. Call GetGeocodingOfLocation first to get coordinates.",
    args_schema= WeatherInput,
)

GetHotelSearchTool = StructuredTool.from_function(
    func=        _search_hotels,
    name=        "GetHotelSearchTool",
    description= "Search hotels near a Sri Lankan destination. Use for: hotels, accommodation, resorts, guesthouses, where to stay, book a room. Call GetGeocodingOfLocation first to get coordinates.",
    args_schema= HotelSearchInput,
)

WebSearchDestinationTool = StructuredTool.from_function(
    func=        _web_search,
    name=        "WebSearchDestinationTool",
    description= "Search for general Sri Lanka travel info, tips, attractions, things to do. Use when no specific hotel/weather is needed. Do NOT use for emergency queries.",
    args_schema= WebSearchInput,
)

EmergencyData = StructuredTool.from_function(
    func=        _emergency_data,
    name=        "EmergencyData",
    description= "Get Sri Lanka emergency contacts. Use for: police, tourist police, ambulance, hospital, fire, lost passport, theft, embassy, disaster, I need help, emergency, danger, unsafe.",
    args_schema= EmergencyInput,
)

GetTrainScheduleTool = StructuredTool.from_function(
    func=        _get_train_schedule,
    name=        "GetTrainScheduleTool",
    description= "Get Sri Lanka train schedules and ticket prices. Use for: train, railway, train times, train schedule, how to get by train, train fares. Default origin to Colombo if not mentioned.",
    args_schema= TrainScheduleInput,
)

GetBusRouteTool = StructuredTool.from_function(
    func=        _get_bus_route,
    name=        "GetBusRouteTool",
    description= """Get Sri Lanka bus routes, fares, schedules and timings between cities.
    Use for: bus, buses, by bus, bus fare, bus schedule, bus times, bus route,
    how to get by bus, coach, CTB, SLTB, public transport.
    Default origin to Colombo if not mentioned.
    Examples: 'bus to Kandy', 'how to get to Galle by bus', 'bus fare Colombo Ella'""",
    args_schema= BusRouteInput,
)