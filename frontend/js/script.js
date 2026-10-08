let dangerMap;
let currentCircle = null;
let trajectoryLayers = [];

// Global variable to hold the generated XML
let currentCapXml = "";

function initMap() {
    dangerMap = L.map('dangerMap').setView([15.0, 88.0], 4);
    
    // Standard OpenStreetMap tiles - Free, no API Key needed
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 19
    }).addTo(dangerMap);
}

function updateMap(radiusKm, colorCode, lat, lon, prevLat = null, prevLon = null) {
    if (currentCircle !== null) dangerMap.removeLayer(currentCircle);
    trajectoryLayers.forEach(layer => dangerMap.removeLayer(layer));
    trajectoryLayers = [];
    
    const currentCenter = [lat, lon]; 
    
    currentCircle = L.circle(currentCenter, {
        color: colorCode,
        fillColor: colorCode,
        fillOpacity: 0.3,
        radius: radiusKm * 1000 
    }).addTo(dangerMap);

    if (prevLat !== null && prevLon !== null && (prevLat !== lat || prevLon !== lon)) {
        const prevCenter = [prevLat, prevLon];

        const prevCircle = L.circle(prevCenter, {
            color: '#94a3b8',
            fillColor: '#cbd5e1',
            fillOpacity: 0.15,
            dashArray: '5, 5',
            radius: (radiusKm * 0.6) * 1000 
        }).addTo(dangerMap).bindPopup("T-24h Position");
        trajectoryLayers.push(prevCircle);

        const stormTrack = L.polyline([prevCenter, currentCenter], {
            color: '#ef4444',
            weight: 3,
            dashArray: '6, 8',
            opacity: 0.8
        }).addTo(dangerMap);
        trajectoryLayers.push(stormTrack);

        const bounds = L.latLngBounds([prevCenter, currentCenter]).pad(0.3);
        dangerMap.fitBounds(bounds, { padding: [30, 30], maxZoom: 8 });
    } else {
        dangerMap.fitBounds(currentCircle.getBounds(), {
            padding: [20, 20],
            maxZoom: 9
        });
    }
}

let currentMode = 'standard';
let files = { standard: null, vis: null, ir: null, wv: null, ri_current: null, ri_prev: null };

const analyzeBtn = document.getElementById('analyze-btn');
const resultsArea = document.getElementById('results-area');
const statusIndicator = document.getElementById('status-indicator');
const statusText = document.getElementById('status-text');
const statusDot = document.getElementById('status-dot-icon');

window.addEventListener('load', () => {
    initMap();
    setTimeout(() => {
        document.getElementById('loader').style.opacity = '0';
        setTimeout(() => document.getElementById('loader').style.display = 'none', 800);
    }, 1200);
});

function transformToDashboard() {
    const landing = document.getElementById('landing-view');
    const dashboard = document.getElementById('dashboard-view');
    const bgImage = document.getElementById('main-bg');

    // 1. Landing view scales out
    landing.classList.add('transform-out');
    
    // 2. Unblur the background image
    bgImage.classList.remove('bg-blurred');
    bgImage.classList.add('bg-clear');

    setTimeout(() => {
        // 3. Swap the content
        landing.style.display = 'none';
        dashboard.style.display = 'block'; 

        // 4. Force reflow and start the smooth fade/slide-up animation
        void dashboard.offsetWidth; 
        dashboard.classList.add('transform-in');
        
        // 5. Allow map sizing to finish
        setTimeout(() => {
            dangerMap.invalidateSize();
        }, 300);

    }, 1000); // Waits for landing page to disappear before showing dashboard
}

function setMode(mode) {
    currentMode = mode;
    document.getElementById('btn-standard').classList.toggle('active', mode === 'standard');
    document.getElementById('btn-multi').classList.toggle('active', mode === 'multi');
    document.getElementById('btn-ri').classList.toggle('active', mode === 'ri');
    
    document.getElementById('drop-zone-standard').style.display = mode === 'standard' ? 'block' : 'none';
    document.getElementById('drop-zone-multi').style.display = mode === 'multi' ? 'grid' : 'none';
    document.getElementById('drop-zone-ri').style.display = mode === 'ri' ? 'block' : 'none';
    
    document.getElementById('preview-container').style.display = 'none';
    
    // Reset the full-width protocol area
    const protocolArea = document.getElementById('protocol-area');
    if(protocolArea) protocolArea.style.display = 'none';
    
    document.getElementById('ri-alert-banner').style.display = 'none';
    document.getElementById('sachet-widget').style.display = 'none'; 
    analyzeBtn.classList.remove('ready');
    resetStatus();
}

function resetStatus() {
    resultsArea.classList.remove('active');
    
    const protocolArea = document.getElementById('protocol-area');
    if(protocolArea) protocolArea.classList.remove('active');

    statusText.innerText = "STANDBY";
    statusDot.style.animation = "none";
    statusIndicator.style.color = "var(--text-muted)";
    statusIndicator.style.background = "rgba(255, 255, 255, 0.05)";
    statusIndicator.style.borderColor = "rgba(255, 255, 255, 0.1)";
}

function checkReadiness() {
    if (currentMode === 'standard' && files.standard) {
        analyzeBtn.classList.add('ready');
    } else if (currentMode === 'multi' && files.vis && files.ir && files.wv) {
        analyzeBtn.classList.add('ready');
    } else if (currentMode === 'ri' && files.ri_current && files.ri_prev) {
        analyzeBtn.classList.add('ready');
    } else {
        analyzeBtn.classList.remove('ready');
    }
}

document.getElementById('file-standard').addEventListener('change', e => {
    if (e.target.files[0]) {
        files.standard = e.target.files[0];
        const reader = new FileReader();
        reader.onload = event => {
            document.getElementById('preview-image').src = event.target.result;
            document.getElementById('preview-container').style.display = 'block';
            checkReadiness();
        };
        reader.readAsDataURL(files.standard);
    }
});

['vis', 'ir', 'wv'].forEach(band => {
    document.getElementById(`file-${band}`).addEventListener('change', e => {
        if (e.target.files[0]) {
            files[band] = e.target.files[0];
            document.getElementById(`status-${band}`).style.display = 'block';
            
            const reader = new FileReader();
            reader.onload = event => {
                const img = document.getElementById(`preview-${band}`);
                img.src = event.target.result;
                img.style.display = 'block';
            };
            reader.readAsDataURL(files[band]);

            checkReadiness();
        }
    });
});

const riInputs = [
    { id: 'ri-current', key: 'ri_current' },
    { id: 'ri-prev', key: 'ri_prev' }
];

riInputs.forEach(input => {
    document.getElementById(`file-${input.id}`).addEventListener('change', e => {
        if (e.target.files[0]) {
            files[input.key] = e.target.files[0];
            document.getElementById(`status-${input.id}`).style.display = 'block';
            
            const reader = new FileReader();
            reader.onload = event => {
                const img = document.getElementById(`preview-${input.id}`);
                img.src = event.target.result;
                img.style.display = 'block';
                
                const iconBox = document.getElementById(`icon-${input.id}`);
                if (iconBox) iconBox.style.display = 'none';
            };
            reader.readAsDataURL(files[input.key]);

            checkReadiness();
        }
    });
});

async function processImage() {
    if (!analyzeBtn.classList.contains('ready')) return;

    analyzeBtn.innerText = "COMPUTING TENSORS...";
    analyzeBtn.style.opacity = '0.7';
    analyzeBtn.style.pointerEvents = 'none';
    
    statusText.innerText = "PROCESSING";
    statusDot.style.animation = "pulse 1s infinite";
    statusIndicator.style.color = "var(--neon-blue)";
    
    const formData = new FormData();
    
    if (currentMode === 'standard') {
        formData.append("file", files.standard);
    } else if (currentMode === 'multi') {
        formData.append("vis_channel", files.vis);
        formData.append("ir_channel", files.ir);
        formData.append("wv_channel", files.wv);
    } else if (currentMode === 'ri') {
        formData.append("file", files.ri_current); 
        formData.append("file_previous", files.ri_prev); 
    }

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error("Backend server error");
        
        const data = await response.json();
        
        if (data.status === "invalid_image") {
            alert("COMMAND ERROR: " + data.message);
            resetStatus();
            return;
        }

        document.getElementById('res-knots').innerText = data.metrics.wind_speed_knots;
        document.getElementById('res-kmh').innerText = data.metrics.wind_speed_kmh;
        document.getElementById('res-pressure').innerText = data.metrics.estimated_pressure_hpa;
        
        document.getElementById('res-imd').innerText = data.classification.imd_scale;
        document.getElementById('res-ss').innerText = data.classification.saffir_simpson;
        
        // --- DYNAMIC XAI GALLERY INJECTION ---
        const gallery = document.getElementById('xaiGallery');
        gallery.innerHTML = ''; // Clear previous heatmaps
        
        if (data.visuals && data.visuals.xai_heatmaps) {
            data.visuals.xai_heatmaps.forEach(heatmapObj => {
                const itemDiv = document.createElement('div');
                itemDiv.className = 'xai-item';
                
                const labelSpan = document.createElement('span');
                labelSpan.className = 'xai-label';
                labelSpan.innerText = heatmapObj.label;
                
                const imgElement = document.createElement('img');
                imgElement.src = heatmapObj.base64;
                
                itemDiv.appendChild(labelSpan);
                itemDiv.appendChild(imgElement);
                gallery.appendChild(itemDiv);
            });
        }

        const threatZone = data.spatial_threat_zone;
        updateMap(
            threatZone.evacuation_radius_km, 
            threatZone.zone_color,
            threatZone.navic_latitude,
            threatZone.navic_longitude,
            threatZone.previous_latitude,
            threatZone.previous_longitude
        );

        const riBanner = document.getElementById('ri-alert-banner');
        if (data.metrics.rapid_intensification_detected) {
            document.getElementById('ri-delta').innerText = data.metrics.intensification_delta_knots;
            riBanner.style.display = 'block';
        } else {
            riBanner.style.display = 'none';
        }

        // --- FULL WIDTH PROTOCOL BINDING ---
        const protocol = data.threat_analysis.advisory_protocol;
        if(protocol) {
            document.getElementById('adv-broadcast').innerText = protocol.official_broadcast;
            document.getElementById('adv-dossier').innerText = protocol.sector_dossier;
            
            const vectorsList = document.getElementById('adv-vectors');
            vectorsList.innerHTML = '';
            protocol.critical_risk_vectors.forEach(risk => {
                let li = document.createElement('li');
                li.innerText = risk;
                vectorsList.appendChild(li);
            });

            const scheduleContainer = document.getElementById('adv-schedule');
            scheduleContainer.innerHTML = '';
            protocol.risk_management_schedule.forEach(step => {
                let item = document.createElement('div');
                item.className = 'timeline-item';
                item.innerHTML = `
                    <div class="timeline-phase">${step.phase}</div>
                    <div class="timeline-action">${step.action}</div>
                `;
                scheduleContainer.appendChild(item);
            });
            
            // Show the new full-width container
            const protocolArea = document.getElementById('protocol-area');
            if(protocolArea) {
                protocolArea.style.display = 'block';
            }
        }

        // --- SACHET XML UI HANDLER ---
        const sachetWidget = document.getElementById('sachet-widget');
        if (data.integrations && data.integrations.sachet_cap_xml) {
            currentCapXml = data.integrations.sachet_cap_xml;
            sachetWidget.style.display = 'block';
            
            const deployBtn = document.getElementById("deployBtn");
            if (deployBtn) {
                deployBtn.innerText = "🚀 CONFIRM DEPLOYMENT";
                deployBtn.style.background = "var(--alert-red)";
                deployBtn.disabled = false;
            }
            
            const widgetBtn = document.querySelector('#sachet-widget button');
            if (widgetBtn) {
                widgetBtn.innerText = "⚠️ REVIEW & TRANSMIT";
                widgetBtn.style.background = "rgba(220, 38, 38, 0.2)";
                widgetBtn.style.borderColor = "var(--alert-red)";
                widgetBtn.style.color = "#fca5a5";
            }
        } else {
            currentCapXml = "";
            if (sachetWidget) sachetWidget.style.display = 'none';
        }

        resultsArea.classList.add('active');
        statusText.innerText = "COMPLETED";
        statusDot.style.animation = "none";
        statusIndicator.style.color = "var(--success-green)";
        statusIndicator.style.background = "rgba(34, 197, 94, 0.1)";
        statusIndicator.style.borderColor = "rgba(34, 197, 94, 0.3)";
        
        // Add active class to fade in the bottom protocol box smoothly
        setTimeout(() => {
            const protocolArea = document.getElementById('protocol-area');
            if(protocolArea) protocolArea.classList.add('active');
            dangerMap.invalidateSize();
        }, 100);
        
    } catch (error) {
        alert("CRITICAL ERROR: Unable to establish link with neural core.");
        console.error(error);
        resetStatus();
    } finally {
        analyzeBtn.innerText = "EXECUTE ANALYSIS";
        analyzeBtn.style.opacity = '1';
        analyzeBtn.style.pointerEvents = 'all';
    }
}

// --- CAP XML MODAL & DEPLOYMENT CONTROLLERS ---

function openCapModal() {
    if (!currentCapXml) return;
    const formattedXml = currentCapXml.replace(/</g, "&lt;").replace(/>/g, "&gt;");
    document.getElementById("capModalCode").innerHTML = formattedXml;
    document.getElementById("capModal").style.display = "flex";
}

function closeCapModal() {
    document.getElementById("capModal").style.display = "none";
}

function copyCapXml() {
    navigator.clipboard.writeText(currentCapXml).then(() => {
        alert("CAP v1.2 XML payload copied to clipboard!");
    });
}

function downloadCapXml() {
    if (!currentCapXml) return;
    const blob = new Blob([currentCapXml], { type: "application/xml" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `NDMA_SACHET_ALERT_${Date.now()}.xml`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

async function deployToWebhook() {
    const deployBtn = document.getElementById("deployBtn");
    const widgetBtn = document.querySelector('#sachet-widget button');
    
    deployBtn.innerText = "TRANSMITTING VIA BACKEND...";
    deployBtn.style.opacity = "0.7";
    deployBtn.disabled = true;

    try {
        // Send the payload to your Python FastAPI server
        const response = await fetch("http://127.0.0.1:8000/transmit", {
            method: "POST",
            headers: { 
                "Content-Type": "application/json" 
            },
            body: JSON.stringify({ xml_data: currentCapXml })
        });

        const data = await response.json();

        if (data.status === "success") {
            deployBtn.innerText = "DEPLOYED SUCCESSFULLY";
            deployBtn.style.background = "var(--success-green)";
            deployBtn.style.opacity = "1";
            
            if (widgetBtn) {
                widgetBtn.innerText = "✓ TRANSMITTED";
                widgetBtn.style.background = "rgba(34, 197, 94, 0.2)";
                widgetBtn.style.borderColor = "var(--success-green)";
                widgetBtn.style.color = "var(--success-green)";
            }
        } else {
            throw new Error(data.message);
        }
        
    } catch (error) {
        console.error("Transmission error:", error);
        alert("CRITICAL: Backend Gateway Error. " + error.message);
        
        deployBtn.innerText = "🚀 CONFIRM DEPLOYMENT";
        deployBtn.style.opacity = "1";
        deployBtn.disabled = false;
    }
}