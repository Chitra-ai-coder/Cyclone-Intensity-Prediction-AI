from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ExifTags
import io
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import uuid

# Import the LLM Intelligence Module
try:
    from backend.llm_intel import get_storm_advisory
except ImportError:
    from llm_intel import get_storm_advisory

app = FastAPI(title="Cyclone AI Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load YOLO model for automated geospatial bounding & cropping
try:
    yolo_model = YOLO("yolov8n.pt")
    print("Successfully loaded YOLOv8 model for automated cropping.")
except Exception as e:
    print(f"Warning: YOLOv8 failed to load. Falling back to center-crop mode: {e}")
    yolo_model = None

# 2. Main Regression Model Architecture
class CycloneIntensityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        return self.resnet(x)

# Load Custom Trained Model with dynamic path resolution
model = CycloneIntensityModel()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(BASE_DIR, "cyclone_resnet_weights.pth")

try:
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Successfully loaded {weights_path}")
except Exception as e:
    print(f"Error loading weights: {e}")
model.to(device)
model.eval()

# 3. Gatekeeper: Lightweight ImageNet Classifier
gatekeeper = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).to(device)
gatekeeper.eval()

# Preprocessing Pipelines
transform_cyclone = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

transform_gatekeeper = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- NavIC / GPS METADATA EXTRACTION PIPELINE ---
def get_decimal_from_dms(dms, ref):
    """Converts Degrees, Minutes, Seconds (DMS) coordinates to Decimal format."""
    degrees = float(dms[0])
    minutes = float(dms[1])
    seconds = float(dms[2])
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def extract_lat_lon(image_bytes: bytes):
    """Robustly parses EXIF headers to extract live NavIC/GPS coordinates from the uploaded satellite feed."""
    default_lat, default_lon = 16.5, 87.5  # Fallback: Central Bay of Bengal
    if not image_bytes:
        return default_lat, default_lon
        
    try:
        original_img = Image.open(io.BytesIO(image_bytes))
        exif = original_img.getexif()
        if not exif:
            return default_lat, default_lon
        
        # Check standard IFD and legacy GPS tag 34853 (0x8825)
        gps_info = exif.get_ifd(ExifTags.IFD.GPSInfo) or exif.get(34853)
        if not gps_info:
            return default_lat, default_lon

        gps_lat = gps_info.get(2)
        gps_lat_ref = gps_info.get(1)
        gps_lon = gps_info.get(4)
        gps_lon_ref = gps_info.get(3)

        if gps_lat and gps_lat_ref and gps_lon and gps_lon_ref:
            # Decode byte references if present (e.g. b'N' -> 'N')
            if isinstance(gps_lat_ref, bytes): gps_lat_ref = gps_lat_ref.decode('utf-8', errors='ignore')
            if isinstance(gps_lon_ref, bytes): gps_lon_ref = gps_lon_ref.decode('utf-8', errors='ignore')
            
            lat = get_decimal_from_dms(gps_lat, gps_lat_ref)
            lon = get_decimal_from_dms(gps_lon, gps_lon_ref)
            return round(lat, 4), round(lon, 4)
    except Exception as e:
        print(f"EXIF extraction error: {e}")
        
    return default_lat, default_lon

def auto_crop_storm(image_bytes: bytes) -> bytes:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    cropped_img = None
    
    if yolo_model is not None:
        try:
            results = yolo_model(img, verbose=False)
            boxes = results[0].boxes
            if len(boxes) > 0:
                box = boxes[0].xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                margin = int(max(x2 - x1, y2 - y1) * 0.25)
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                cropped_img = img[y1:y2, x1:x2]
        except Exception:
            pass

    if cropped_img is None or cropped_img.size == 0:
        min_dim = min(h, w)
        start_x, start_y = (w - min_dim) // 2, (h - min_dim) // 2
        cropped_img = img[start_y:start_y+min_dim, start_x:start_x+min_dim]

    success, encoded_img = cv2.imencode('.jpg', cropped_img)
    return encoded_img.tobytes() if success else image_bytes

# --- NEW: MULTI-STAGE GATEKEEPER ---
# Everyday COCO classes that must NEVER pass as satellite imagery
# 0 = person, 1 = bicycle, 2 = car, 3 = motorcycle, 15 = cat, 16 = dog, 63 = laptop, 67 = cell phone
PROHIBITED_COCO_CLASSES = {0, 1, 2, 3, 5, 7, 15, 16, 56, 57, 60, 62, 63, 67}

def is_valid_satellite_candidate(image_bytes: bytes, pil_image: Image.Image) -> bool:
    """
    Two-stage gatekeeper:
    1. Rejects if YOLOv8 explicitly detects humans, pets, or everyday items.
    2. Uses MobileNetV2 to catch other common terrestrial objects.
    """
    # Stage 1: Explicit Object Detection with YOLO
    if yolo_model is not None:
        try:
            np_arr = np.frombuffer(image_bytes, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            results = yolo_model(cv_img, verbose=False)
            
            for box in results[0].boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                
                # If a human or domestic object is detected with > 30% confidence, reject immediately
                if class_id in PROHIBITED_COCO_CLASSES and confidence > 0.30:
                    print(f"[Gatekeeper] Rejected: Detected COCO class {class_id} (conf: {confidence:.2f})")
                    return False
        except Exception as e:
            print(f"[Gatekeeper] YOLO verification error: {e}")

    # Stage 2: MobileNet fallback check for non-satellite textures
    tensor = transform_gatekeeper(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = gatekeeper(tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_idx = torch.topk(probabilities, 1)

    # Rejection threshold: Lowered to 0.22 to catch low-confidence everyday scenes
    if top_prob.item() > 0.22:
        print(f"[Gatekeeper] Rejected by MobileNet: Class {top_idx.item()} (prob: {top_prob.item():.2f})")
        return False

    return True
# -----------------------------------

def calculate_danger_zone(knots: float):
    if knots < 34: return {"evacuation_radius_km": 45, "zone_color": "#38bdf8"}
    elif knots < 64: return {"evacuation_radius_km": 90, "zone_color": "#facc15"}
    elif knots < 90: return {"evacuation_radius_km": 140, "zone_color": "#fb923c"}
    elif knots < 120: return {"evacuation_radius_km": 200, "zone_color": "#f87171"}
    else: return {"evacuation_radius_km": 280, "zone_color": "#dc2626"}

def categorize_storm(knots: float):
    if knots < 28: imd = "Depression"
    elif knots < 34: imd = "Deep Depression"
    elif knots < 48: imd = "Cyclonic Storm"
    elif knots < 64: imd = "Severe Cyclonic Storm"
    elif knots < 90: imd = "Very Severe Cyclonic Storm"
    elif knots < 120: imd = "Extremely Severe Cyclonic Storm"
    else: imd = "Super Cyclonic Storm"

    if knots < 34: ss = "Tropical Depression"
    elif knots < 64: ss = "Tropical Storm"
    elif knots < 83: ss = "Category 1 Hurricane"
    elif knots < 96: ss = "Category 2 Hurricane"
    elif knots < 113: ss = "Category 3 Major Hurricane"
    elif knots < 137: ss = "Category 4 Major Hurricane"
    else: ss = "Category 5 Major Hurricane"

    if knots < 48: risk, warning = "Low", "Minor structural damage, localized power outages."
    elif knots < 83: risk, warning = "Moderate", "Considerable roof damage, large trees uprooted, coastal sea flooding."
    elif knots < 113: risk, warning = "High", "Extensive structural collapse, severe storm surges, major communication loss."
    else: risk, warning = "Catastrophic", "Complete structural destruction, widespread devastating storm surge, uninhabitable coast."

    return imd, ss, risk, warning

def generate_xai_attention_map(input_tensor, original_image):
    activation = {}
    def get_activation(name):
        def hook(model, input, output): activation[name] = output.detach()
        return hook
        
    handle = model.resnet.layer4.register_forward_hook(get_activation('layer4'))
    with torch.no_grad(): _ = model(input_tensor)
    handle.remove()
    
    act = activation['layer4'].squeeze()
    heatmap = torch.mean(act, dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0: heatmap /= np.max(heatmap)
        
    heatmap = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, (224, 224))), cv2.COLORMAP_JET)
    original_resized = cv2.cvtColor(cv2.resize(np.array(original_image), (224, 224)), cv2.COLOR_RGB2BGR)
    
    superimposed_img = cv2.cvtColor(cv2.addWeighted(original_resized, 0.5, heatmap, 0.5, 0), cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(superimposed_img)
    plt.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

# --- CAP v1.2 XML GENERATOR ---
def generate_cap_xml(lat: float, lon: float, radius: float, imd_scale: str, risk_level: str, advisory: dict) -> str:
    """Generates an NDMA SACHET compliant CAP v1.2 XML string."""
    alert = ET.Element("alert", xmlns="urn:oasis:names:tc:emergency:cap:1.2")
    
    ET.SubElement(alert, "identifier").text = f"NDMA-AI-{uuid.uuid4().hex[:8]}"
    ET.SubElement(alert, "sender").text = "eoc-automated@sachet.ndma.gov.in"
    ET.SubElement(alert, "sent").text = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    ET.SubElement(alert, "status").text = "Actual" 
    ET.SubElement(alert, "msgType").text = "Alert"
    ET.SubElement(alert, "scope").text = "Public"
    
    info = ET.SubElement(alert, "info")
    ET.SubElement(info, "category").text = "Met"
    ET.SubElement(info, "event").text = f"Cyclone - {imd_scale}"
    ET.SubElement(info, "responseType").text = "Evacuate"
    
    ET.SubElement(info, "urgency").text = "Immediate"
    ET.SubElement(info, "severity").text = "Extreme" if risk_level in ["High", "Catastrophic"] else "Severe"
    ET.SubElement(info, "certainty").text = "Observed"
    
    ET.SubElement(info, "headline").text = advisory.get("official_broadcast", "Automated Cyclone Alert")
    ET.SubElement(info, "description").text = " | ".join(advisory.get("critical_risk_vectors", []))
    
    area = ET.SubElement(info, "area")
    ET.SubElement(area, "areaDesc").text = "Maritime and Coastal Sectors"
    ET.SubElement(area, "circle").text = f"{lat},{lon} {radius}"
    
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(alert, encoding="unicode")

@app.post("/predict")
async def predict_cyclone(
    file: Optional[UploadFile] = File(None),
    file_previous: Optional[UploadFile] = File(None), 
    vis_channel: Optional[UploadFile] = File(None),
    ir_channel: Optional[UploadFile] = File(None),
    wv_channel: Optional[UploadFile] = File(None)
):
    if vis_channel and ir_channel and wv_channel:
        vis = Image.open(io.BytesIO(await vis_channel.read())).convert("L")
        ir = Image.open(io.BytesIO(await ir_channel.read())).convert("L").resize(vis.size)
        wv = Image.open(io.BytesIO(await wv_channel.read())).convert("L").resize(vis.size)
        fused_satellite = Image.merge("RGB", (vis, ir, wv))
        img_buffer = io.BytesIO()
        fused_satellite.save(img_buffer, format="JPEG")
        image_bytes = img_buffer.getvalue()
    elif file:
        image_bytes = await file.read()
    else:
        return {"status": "error", "message": "No telemetry payload supplied."}
    
    # 1. Extract NavIC/GPS coordinates for CURRENT image
    lat, lon = extract_lat_lon(image_bytes)
    
    # 2. Automated Crop
    processed_bytes = auto_crop_storm(image_bytes)
    raw_image = Image.open(io.BytesIO(processed_bytes)).convert("RGB")
    
    # 3. Gatekeeper (UPDATED for Two-Stage Checks)
    if not is_valid_satellite_candidate(image_bytes, raw_image):
        return {"status": "invalid_image", "message": "Invalid input: Terrestrial or non-satellite subject detected."}
    
    # 4. Grayscale normalization for ResNet50
    cyclone_img = raw_image.convert("L").convert("RGB")
    input_tensor = transform_cyclone(cyclone_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor).item()
        predicted_knots = round(prediction, 1)
    
    # 5. Sanity Check
    if predicted_knots < 0 or predicted_knots > 200:
        return {"status": "invalid_image", "message": "Unrecognized pattern."}
    
    # Initialize array to store generated heatmaps
    generated_heatmaps = []

    # Generate XAI Heatmap for Current Image (T-0)
    current_heatmap_b64 = generate_xai_attention_map(input_tensor, raw_image)
    generated_heatmaps.append({
        "label": "T-0 (Current Telemetry)",
        "base64": current_heatmap_b64
    })

    # --- RAPID INTENSIFICATION (RI) & TRAJECTORY TRACKER ---
    rapid_intensification_flag = False
    delta_knots = 0
    prev_lat, prev_lon = None, None
    
    if file_previous:
        try:
            prev_bytes = await file_previous.read()
            
            # Extract NavIC/GPS coordinates for PREVIOUS image
            prev_lat, prev_lon = extract_lat_lon(prev_bytes)
            
            prev_processed = auto_crop_storm(prev_bytes)
            prev_raw = Image.open(io.BytesIO(prev_processed)).convert("RGB")
            prev_img = prev_raw.convert("L").convert("RGB")
            prev_tensor = transform_cyclone(prev_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                previous_knots = round(model(prev_tensor).item(), 1)
            
            # Generate XAI Heatmap for Previous Image (T-24H)
            prev_heatmap_b64 = generate_xai_attention_map(prev_tensor, prev_raw)
            generated_heatmaps.append({
                "label": "T-24H (Previous Telemetry)",
                "base64": prev_heatmap_b64
            })

            delta_knots = round(predicted_knots - previous_knots, 1)
            # Meteorological definition of RI: >= 30 knot jump
            if delta_knots >= 30:
                rapid_intensification_flag = True
        except Exception as e:
            print(f"Error processing RI previous frame: {e}")

    kmh = round(predicted_knots * 1.852, 1)
    pressure = round(1010 - ((predicted_knots / 14.0) ** 2), 1)
    
    imd, ss, risk, warning = categorize_storm(predicted_knots)
    spatial_threat = calculate_danger_zone(predicted_knots)

    # Determine if Multi-Spectral mode is active
    is_multi = True if (vis_channel and ir_channel and wv_channel) else False

    # 6. Generate AI Disaster Management Advisory via Gemini LLM
    advisory_data = get_storm_advisory(
        lat=lat,
        lon=lon,
        knots=predicted_knots,
        kmh=kmh,
        pressure=pressure,
        imd_scale=imd,
        ss_scale=ss,
        radius=spatial_threat["evacuation_radius_km"],
        is_multispectral=is_multi,
        delta_knots=delta_knots if file_previous else None,
        prev_lat=prev_lat,
        prev_lon=prev_lon
    )

    # --- SACHET XML AUTO-GENERATION TRIGGER ---
    CRITICAL_ALERT_THRESHOLD = 34.0
    cap_xml_payload = None

    if predicted_knots >= CRITICAL_ALERT_THRESHOLD:
        cap_xml_payload = generate_cap_xml(
            lat=lat,
            lon=lon,
            radius=spatial_threat["evacuation_radius_km"],
            imd_scale=imd,
            risk_level=risk,
            advisory=advisory_data
        )

    return {
        "status": "success",
        "metrics": {
            "wind_speed_knots": predicted_knots,
            "wind_speed_kmh": kmh,
            "estimated_pressure_hpa": pressure,
            "rapid_intensification_detected": rapid_intensification_flag,
            "intensification_delta_knots": delta_knots
        },
        "classification": {
            "saffir_simpson": ss,
            "imd_scale": imd
        },
        "threat_analysis": {
            "risk_level": risk,
            "impact_warning": warning,
            "advisory_protocol": advisory_data
        },
        "spatial_threat_zone": {
            "evacuation_radius_km": spatial_threat["evacuation_radius_km"],
            "zone_color": spatial_threat["zone_color"],
            "navic_latitude": lat,
            "navic_longitude": lon,
            "previous_latitude": prev_lat,
            "previous_longitude": prev_lon
        },
        "visuals": {
            "xai_heatmaps": generated_heatmaps
        },
        "integrations": {
            "sachet_cap_xml": cap_xml_payload
        }
    }

# --- NEW: SECURE TRANSMISSION GATEWAY ---
class TransmitPayload(BaseModel):
    xml_data: str

@app.post("/transmit")
async def transmit_alert(payload: TransmitPayload):
    # Your specific Webhook URL from the screenshot
    webhook_url = "https://webhook.site/20157cc3-bb2a-49cd-a622-a3903ebf1b35"
    headers = {"Content-Type": "application/xml"}
    
    try:
        # Python 'requests' bypasses browser CORS entirely
        response = requests.post(webhook_url, data=payload.xml_data, headers=headers)
        if response.status_code == 200:
            return {"status": "success"}
        else:
            return {"status": "error", "message": f"Gateway returned {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}