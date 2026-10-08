import os
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Automatically load .env from the project root directory
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Initialize Gemini Client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Warning: GEMINI_API_KEY is not set in environment or .env file.")

client = genai.Client(api_key=api_key) if api_key else None

def get_storm_advisory(
    lat: float, 
    lon: float, 
    knots: float, 
    kmh: float, 
    pressure: float, 
    imd_scale: str, 
    ss_scale: str, 
    radius: int,
    is_multispectral: bool = False,
    delta_knots: float = None,
    prev_lat: float = None,
    prev_lon: float = None
) -> dict:
    """
    Queries Gemini 2.5 Flash to generate a structured meteorological intelligence
    dossier, dynamically adapting to Multi-Spectral or Time-Series RI modes.
    """
    fallback_response = {
        "official_broadcast": f"ALERT: {imd_scale} detected near NavIC coordinates {lat}°N, {lon}°E with sustained winds of {kmh} km/h. Coastal sectors must initiate immediate precautionary protocols.",
        "sector_dossier": f"NavIC Coordinates {lat}°N, {lon}°E lie in a high-risk maritime tracking corridor. Regional vulnerability requires active surveillance.",
        "critical_risk_vectors": [
            f"High-velocity destructive wind gusts reaching {kmh} km/h.",
            f"Central atmospheric pressure drop to {pressure} hPa indicating rapid intensification.",
            f"Mandatory evacuation perimeter established at {radius} km radius."
        ],
        "risk_management_schedule": [
            {"phase": "Immediate (T-0 to T-12 hrs)", "action": "Suspend maritime operations and deploy NDRF advance teams."},
            {"phase": "Short-term (T-12 to T-24 hrs)", "action": f"Enforce mandatory evacuation across the {radius} km hazard radius."},
            {"phase": "Post-Landfall", "action": "Deploy SDRF structural damage assessment units and restore grids."}
        ]
    }

    if not client:
        return fallback_response

    # --- DYNAMIC CONTEXT INJECTION ---
    mode_context = "MODE: STANDARD RGB SATELLITE TELEMETRY"
    
    if is_multispectral:
        mode_context = """
        MODE: MULTI-SPECTRAL SENSOR FUSION ACTIVE
        Telemetry includes Visible (VIS), Infrared (IR), and Water Vapor (WV) channels. 
        INSTRUCTION: In your 'sector_dossier' and 'critical_risk_vectors', elaborately explain what these multi-spectral bands reveal (e.g., deep convective tower structures in IR, upper-level moisture envelopment in WV, and low-level circulation in VIS). Provide deep technical meteorological insights on the storm's physical structure based on multi-band data.
        """
    elif delta_knots is not None and prev_lat is not None and prev_lon is not None:
        ri_status = "CRITICAL RAPID INTENSIFICATION DETECTED" if delta_knots >= 30 else "Standard Intensification Phase"
        mode_context = f"""
        MODE: TIME-SERIES TRAJECTORY & INTENSIFICATION TRACKING ACTIVE
        - Previous NavIC Coordinates (T-24h): Lat {prev_lat}, Lon {prev_lon}
        - Current NavIC Coordinates (T-0h): Lat {lat}, Lon {lon}
        - 24hr Wind Speed Delta: +{delta_knots} knots ({ri_status})
        INSTRUCTION: In your 'sector_dossier', specifically analyze the storm's geographic movement trajectory from the previous to current coordinates. Emphasize the severe risks and accelerated timeline associated with its rate of intensification (+{delta_knots} knots).
        """

    prompt = f"""
    You are an authoritative Disaster Management AI operating under the guidelines of the Indian National Disaster Management Authority (NDMA).
    A cyclonic system has been intercepted via NavIC/GPS telemetry.

    {mode_context}

    TELEMETRY & THREAT PARAMETERS:
    - Current NavIC Coordinates: Latitude {lat}, Longitude {lon}
    - Wind Velocity: {knots} knots ({kmh} km/h)
    - Minimum Central Pressure: {pressure} hPa
    - IMD Scale: {imd_scale}
    - Danger Zone Radius: {radius} km

    Produce an official NDMA-compliant disaster management protocol. 
    Return ONLY a single valid JSON object strictly matching this exact schema without any markdown formatting:
    {{
      "official_broadcast": "A 2-sentence urgent public warning (incorporate NDMA authority tone) suitable for SMS.",
      "sector_dossier": "Historical briefing of past cyclones hitting these specific NavIC coordinates, combined with the specific analytical instructions from the active telemetry MODE above.",
      "critical_risk_vectors": [
        "Primary NDMA threat vector 1 (tailored to the active telemetry MODE)",
        "Secondary NDMA threat vector 2",
        "Tertiary NDMA threat vector 3"
      ],
      "risk_management_schedule": [
        {{"phase": "Immediate (T-0 to T-12 hrs)", "action": "NDMA NDRF deployment directive."}},
        {{"phase": "Short-term (T-12 to T-24 hrs)", "action": "State/District administration evacuation protocol."}},
        {{"phase": "Post-Landfall", "action": "SDRF relief and grid restoration directive."}}
      ]
    }}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3
            )
        )
        if response.text:
            return json.loads(response.text)
        return fallback_response
    except Exception as e:
        print(f"LLM Intelligence Error: {e}")
        return fallback_response