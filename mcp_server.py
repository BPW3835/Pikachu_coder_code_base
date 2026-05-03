# from fastmcp import FastMCP

# # Initialize the MCP server with a descriptive name
# mcp = FastMCP("mcp-server")

# @mcp.tool
# def get_weather(city: str) -> dict:
#     """Get the current weather for a city."""
#     # In production, this would call a real weather API
#     weather_data = {
#         "delhi": {"temp": 32, "condition": "sunny"},
#         "swindon": {"temp": 45, "condition": "cloudy"},
#         "tokyo": {"temp": -1, "condition": "snowfall"},
#     }

#     city_lower = city.lower()
#     if city_lower in weather_data:
#         return {"city": city, **weather_data[city_lower]}
#     else:
#         return {"city": city, "temp": 70, "condition": "unknown"}

# if __name__ == "__main__":
#     # Run the server over stdio for local development
#     mcp.run(transport="stdio")
    
    
import os
import requests
from fastmcp import FastMCP

# -------------------------------------------------
# MCP Server Initialization
# -------------------------------------------------
mcp = FastMCP("my-first-server")

BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


@mcp.tool
def get_weather(city: str) -> dict:
    """
    Get the current weather for a city using OpenWeatherMap API.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return {
            "error": "OPENWEATHER_API_KEY not configured",
            "hint": "Pass it via MCP client env or set it in the runtime environment",
        }

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        return {
            "city": data.get("name", city),
            "temperature_celsius": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "condition": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"],
        }

    except requests.exceptions.HTTPError:
        # Most commonly: 404 city not found
        return {"error": "City not found"}
    except requests.exceptions.RequestException as e:
        # Network / timeout / other request issues
        return {"error": str(e)}


if __name__ == "__main__":
    # stdio is best for local development and agent frameworks
    mcp.run(transport="stdio")