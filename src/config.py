from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "landslide_rfc_model.pkl"

# Model Expected Input Contract (11 Features)
INPUT_COLUMNS = [
    "Geology_Type", "Land_Use", "Slope_Angle", "Rainfall_3Day", 
    "Effective_Rainfall_mm", "Soil_Saturation", "Vegetation_Cover", 
    "Elevation_m", "Soil_Erosion_Rate", "Pore_Pressure_Ratio", "Seismic_PGA_g"
]

CATEGORICAL_OPTIONS = {
    "Geology_Type": ["Colluvium", "Igneous_Metamorphic", "Residual_Soil", "Sedimentary_Rock", "Weathered_Rock"],
    "Land_Use": ["Agriculture", "Barren", "Forest", "Urban"]
}

# Interface ranges (low, high, default, step, unit, format)
NUMERIC_INPUTS = {
    "Slope_Angle": (0.0, 90.0, 35.0, 1.0, "°", "%.0f"),
    "Rainfall_3Day": (0.0, 500.0, 150.0, 5.0, "mm", "%.0f"),
    "Effective_Rainfall_mm": (0.0, 500.0, 100.0, 5.0, "mm", "%.0f"),
    "Soil_Saturation": (0.0, 1.0, 0.65, 0.01, "", "%.2f"),
    "Vegetation_Cover": (0.0, 1.0, 0.50, 0.01, "", "%.2f"),
    "Elevation_m": (0.0, 5000.0, 1200.0, 10.0, "m", "%.0f"),
    "Soil_Erosion_Rate": (0.0, 20.0, 4.0, 0.1, "", "%.1f"),
    "Pore_Pressure_Ratio": (0.0, 1.0, 0.45, 0.01, "", "%.2f"),
    "Seismic_PGA_g": (0.0, 1.0, 0.20, 0.01, "g", "%.2f"),
}

# Logical Control Groupings
CONTROL_GROUPS = (
    ("Site Location", ("Latitude", "Longitude")),
    ("Site Characteristics", ("Geology_Type", "Land_Use", "Elevation_m")),
    ("Seismic", ("Seismic_PGA_g",)),
    ("Terrain", ("Slope_Angle", "Soil_Erosion_Rate", "Vegetation_Cover")),
    ("Hydrology", ("Rainfall_3Day", "Soil_Saturation", "Pore_Pressure_Ratio")),
)

# Strategic Lifeline Highway Corridors for Action Matrix
HIGHWAY_CORRIDORS = {
    "Gangtok": {"highway": "NH-10 (Sevoke-Gangtok)", "lifeline_for": "Sikkim Strategic Military & Civil Lifeline"},
    "Darjeeling": {"highway": "NH-110 (Siliguri-Darjeeling)", "lifeline_for": "Tea & Tourism Arterial Corridor"},
    "Aizawl": {"highway": "NH-54 (Silchar-Aizawl)", "lifeline_for": "Mizoram Fuel & Food Grain Supply Line"},
    "Kohima": {"highway": "NH-29 (Dimapur-Kohima-Imphal)", "lifeline_for": "Nagaland-Manipur Sole Logistics Line"},
    "Shillong": {"highway": "NH-6 (Jorabat-Shillong-Silchar)", "lifeline_for": "Meghalaya & Barak Valley Transit"},
    "Guwahati": {"highway": "NH-27 (East-West Corridor)", "lifeline_for": "Gateway to All 7 NE States"},
    "Cherrapunji (Sohra)": {"highway": "SH-5 (Shillong-Cherra)", "lifeline_for": "High-Rainfall Tourist & Rural Link"},
    "Tawang": {"highway": "NH-13 (Trans-Arunachal Highway)", "lifeline_for": "Border Defence Logistics Arterial"},
    "Dima Hasao (Haflong)": {"highway": "NH-54E (Lumding-Silchar)", "lifeline_for": "Hill Section Railway & Freight Pass"},
    "Imphal": {"highway": "NH-37 (Silchar-Jiribam-Imphal)", "lifeline_for": "Manipur Alternate Lifeline Corridor"},
    "NH-10 Sevoke Choke": {"highway": "NH-10 (29th Mile Choke)", "lifeline_for": "Teesta River Flood Choke Point"},
    "NH-29 Pagala Pahar": {"highway": "NH-29 (Pagala Pahar Sinking Zone)", "lifeline_for": "Active Landslide Failure Belt"},
    "NH-54 Kawnpui Pass": {"highway": "NH-54 (Kawnpui Ridge)", "lifeline_for": "Central Mizoram Arterial Link"},
    "Dibang Gorge": {"highway": "NH-313 (Roing-Anini Corridor)", "lifeline_for": "Eastern Arunachal Valley Lifeline"}
}

# Google Maps-Style Hierarchical Typography & Regional Attributes
NER_GEOGRAPHIC_REGIONS = {
    # TIER 1: Macro-Regions (Large Typography)
    "SIKKIM HIMALAYAS": {"lat": 27.55, "lon": 88.50, "size": 14, "tier": 1, "slope": 46.0, "geo": "Weathered_Rock", "elev": 2800.0, "risk": "High"},
    "BRAHMAPUTRA VALLEY": {"lat": 26.50, "lon": 92.50, "size": 15, "tier": 1, "slope": 8.0, "geo": "Residual_Soil", "elev": 95.0, "risk": "Low"},
    "MEGHALAYA PLATEAU": {"lat": 25.50, "lon": 91.30, "size": 13, "tier": 1, "slope": 32.0, "geo": "Sedimentary_Rock", "elev": 1400.0, "risk": "High"},
    "NAGA HILLS": {"lat": 26.10, "lon": 94.60, "size": 13, "tier": 1, "slope": 38.0, "geo": "Colluvium", "elev": 1650.0, "risk": "High"},
    "MIZO RIDGES": {"lat": 23.30, "lon": 92.80, "size": 13, "tier": 1, "slope": 52.0, "geo": "Weathered_Rock", "elev": 1200.0, "risk": "Extreme"},
    "ARUNACHAL FRONTIER": {"lat": 28.20, "lon": 94.50, "size": 14, "tier": 1, "slope": 48.0, "geo": "Igneous_Metamorphic", "elev": 2400.0, "risk": "High"},

    # TIER 2: Key Cities & Mountain Hubs (Medium Typography)
    "Gangtok": {"lat": 27.33, "lon": 88.61, "size": 12, "tier": 2, "slope": 42.0, "geo": "Weathered_Rock", "elev": 1650.0, "risk": "High"},
    "Darjeeling": {"lat": 27.03, "lon": 88.26, "size": 12, "tier": 2, "slope": 55.0, "geo": "Residual_Soil", "elev": 2045.0, "risk": "Extreme"},
    "Aizawl": {"lat": 23.73, "lon": 92.71, "size": 12, "tier": 2, "slope": 58.0, "geo": "Colluvium", "elev": 1132.0, "risk": "Extreme"},
    "Kohima": {"lat": 25.67, "lon": 94.10, "size": 12, "tier": 2, "slope": 40.0, "geo": "Residual_Soil", "elev": 1444.0, "risk": "High"},
    "Shillong": {"lat": 25.57, "lon": 91.89, "size": 12, "tier": 2, "slope": 28.0, "geo": "Sedimentary_Rock", "elev": 1525.0, "risk": "Moderate"},
    "Guwahati": {"lat": 26.14, "lon": 91.73, "size": 12, "tier": 2, "slope": 22.0, "geo": "Sedimentary_Rock", "elev": 55.0, "risk": "Low"},
    "Cherrapunji (Sohra)": {"lat": 25.27, "lon": 91.73, "size": 11, "tier": 2, "slope": 38.0, "geo": "Sedimentary_Rock", "elev": 1430.0, "risk": "Extreme"},
    "Tawang": {"lat": 27.58, "lon": 91.86, "size": 11, "tier": 2, "slope": 48.0, "geo": "Weathered_Rock", "elev": 3048.0, "risk": "High"},
    "Dima Hasao (Haflong)": {"lat": 25.17, "lon": 93.02, "size": 11, "tier": 2, "slope": 42.0, "geo": "Colluvium", "elev": 680.0, "risk": "High"},
    "Imphal": {"lat": 24.81, "lon": 93.93, "size": 11, "tier": 2, "slope": 14.0, "geo": "Residual_Soil", "elev": 786.0, "risk": "Low"},

    # TIER 3: Sensitive Highway Chokepoints (Compact Typography)
    "NH-10 Sevoke Choke": {"lat": 27.15, "lon": 88.45, "size": 9, "tier": 3, "slope": 45.0, "geo": "Colluvium", "elev": 450.0, "risk": "High"},
    "NH-29 Pagala Pahar": {"lat": 25.80, "lon": 93.85, "size": 9, "tier": 3, "slope": 44.0, "geo": "Weathered_Rock", "elev": 890.0, "risk": "Extreme"},
    "NH-54 Kawnpui Pass": {"lat": 23.95, "lon": 92.68, "size": 9, "tier": 3, "slope": 50.0, "geo": "Colluvium", "elev": 920.0, "risk": "High"},
    "Dibang Gorge": {"lat": 28.62, "lon": 95.77, "size": 9, "tier": 3, "slope": 50.0, "geo": "Igneous_Metamorphic", "elev": 1800.0, "risk": "High"}
}

# Alias for backward compatibility
MONITORING_STATIONS = NER_GEOGRAPHIC_REGIONS

# 48-Node Spatial Mesh for Continuous GIS Rainbow Density Heatmap
NER_HEATMAP_GRID = [
    {"lat": 27.33, "lon": 88.61, "intensity": 88},
    {"lat": 27.28, "lon": 88.50, "intensity": 72},
    {"lat": 27.03, "lon": 88.26, "intensity": 94},
    {"lat": 27.15, "lon": 88.40, "intensity": 65},
    {"lat": 27.50, "lon": 88.55, "intensity": 58},
    {"lat": 27.58, "lon": 91.86, "intensity": 82},
    {"lat": 27.35, "lon": 92.42, "intensity": 75},
    {"lat": 27.10, "lon": 92.70, "intensity": 55},
    {"lat": 27.95, "lon": 93.10, "intensity": 68},
    {"lat": 28.05, "lon": 94.12, "intensity": 78},
    {"lat": 28.30, "lon": 94.70, "intensity": 60},
    {"lat": 28.62, "lon": 95.77, "intensity": 90},
    {"lat": 28.15, "lon": 95.30, "intensity": 70},
    {"lat": 27.80, "lon": 96.10, "intensity": 64},
    {"lat": 25.27, "lon": 91.73, "intensity": 99},
    {"lat": 25.30, "lon": 91.55, "intensity": 92},
    {"lat": 25.20, "lon": 91.90, "intensity": 86},
    {"lat": 25.57, "lon": 91.89, "intensity": 62},
    {"lat": 25.51, "lon": 90.22, "intensity": 58},
    {"lat": 25.40, "lon": 90.75, "intensity": 52},
    {"lat": 26.14, "lon": 91.73, "intensity": 45},
    {"lat": 26.40, "lon": 90.27, "intensity": 25},
    {"lat": 26.65, "lon": 92.79, "intensity": 35},
    {"lat": 26.58, "lon": 93.17, "intensity": 30},
    {"lat": 27.47, "lon": 94.91, "intensity": 38},
    {"lat": 26.90, "lon": 94.20, "intensity": 32},
    {"lat": 25.17, "lon": 93.02, "intensity": 85},
    {"lat": 24.83, "lon": 92.77, "intensity": 42},
    {"lat": 25.67, "lon": 94.10, "intensity": 95},
    {"lat": 25.75, "lon": 94.25, "intensity": 84},
    {"lat": 26.32, "lon": 94.52, "intensity": 76},
    {"lat": 25.90, "lon": 93.72, "intensity": 40},
    {"lat": 26.10, "lon": 94.00, "intensity": 62},
    {"lat": 26.60, "lon": 94.80, "intensity": 70},
    {"lat": 24.81, "lon": 93.93, "intensity": 35},
    {"lat": 24.33, "lon": 93.67, "intensity": 78},
    {"lat": 25.15, "lon": 94.40, "intensity": 72},
    {"lat": 24.70, "lon": 93.45, "intensity": 80},
    {"lat": 23.73, "lon": 92.71, "intensity": 98},
    {"lat": 23.85, "lon": 92.75, "intensity": 88},
    {"lat": 23.50, "lon": 92.65, "intensity": 82},
    {"lat": 22.88, "lon": 92.74, "intensity": 89},
    {"lat": 22.40, "lon": 92.90, "intensity": 74},
    {"lat": 23.20, "lon": 93.10, "intensity": 70},
    {"lat": 23.83, "lon": 91.28, "intensity": 22},
    {"lat": 24.37, "lon": 92.16, "intensity": 48},
    {"lat": 23.80, "lon": 91.95, "intensity": 40},
]

# GPS Coordinates tracing actual mountain highway passes through river gorges & ridges
HIGHWAY_POLYLINE_COORDINATES = {
    "NH-10 (Siliguri - Gangtok)": {
        "associated_station": "Gangtok",
        "highway_name": "NH-10 Sevoke-Teesta Lifeline",
        "chokepoints": "29th Mile, Kalijhora, Teesta Bazaar, Rangpo",
        "coordinates": [
            [26.7271, 88.3953],  # Siliguri
            [26.8833, 88.4667],  # Sevoke (Coronation Bridge)
            [26.9312, 88.4611],  # Kalijhora
            [27.0514, 88.4623],  # 29th Mile (Frequent Sinking Zone)
            [27.0722, 88.4311],  # Teesta Bazaar
            [27.1767, 88.5300],  # Rangpo Border Checkpost
            [27.2341, 88.5023],  # Singtam
            [27.2941, 88.5912],  # Ranipool
            [27.3314, 88.6138],  # Gangtok
        ]
    },
    "NH-29 (Dimapur - Kohima)": {
        "associated_station": "Kohima",
        "highway_name": "NH-29 Dimapur-Kohima Corridor",
        "chokepoints": "Pagala Pahar Sinking Zone, Chumukedima, Zubza",
        "coordinates": [
            [25.9068, 93.7275],  # Dimapur
            [25.8012, 93.7712],  # Chumukedima Pass
            [25.7511, 93.8423],  # Pagala Pahar (Active Slide Belt)
            [25.7623, 93.8812],  # Medziphema
            [25.7412, 93.9712],  # Piphema
            [25.7012, 94.0321],  # Zubza
            [25.6751, 94.1086],  # Kohima
        ]
    },
    "NH-54 (Silchar - Aizawl)": {
        "associated_station": "Aizawl",
        "highway_name": "NH-54 / NH-306 Mizoram Lifeline",
        "chokepoints": "Kawnpui Ridge, Vairengte Pass, Kolasib",
        "coordinates": [
            [24.8333, 92.7789],  # Silchar Gateway
            [24.6012, 92.8412],  # Dholai
            [24.5123, 92.7612],  # Vairengte Checkpost
            [24.3312, 92.7212],  # Bilkhawthlir
            [24.2312, 92.6812],  # Kolasib Pass
            [23.9512, 92.6812],  # Kawnpui Sinking Ridge
            [23.8112, 92.7312],  # Selesih
            [23.7307, 92.7173],  # Aizawl
        ]
    },
    "NH-6 (Shillong - Silchar)": {
        "associated_station": "Shillong",
        "highway_name": "NH-6 Meghalaya-Barak Valley Lifeline",
        "chokepoints": "Sonapur Tunnel, Lumshnong, Jowai Sinking Zone",
        "coordinates": [
            [25.5788, 91.8933],  # Shillong
            [25.5122, 92.1012],  # Jowai
            [25.3211, 92.3512],  # Lad Rymbai
            [25.1412, 92.4212],  # Sonapur Tunnel (Active Mudslide Sinking Zone)
            [25.0512, 92.5112],  # Lumshnong
            [24.8812, 92.6512],  # Badarpur Ghat
            [24.8333, 92.7789],  # Silchar Gateway
        ]
    },
    "NH-13 (Bhalukpong - Tawang)": {
        "associated_station": "Tawang",
        "highway_name": "NH-13 Trans-Arunachal Defense Highway",
        "chokepoints": "Sela Pass, Tenga Valley, Bomdila Ridge",
        "coordinates": [
            [27.0112, 92.6512],  # Bhalukpong Gate
            [27.1812, 92.5512],  # Tenga Valley
            [27.2612, 92.4212],  # Bomdila Pass
            [27.3512, 92.2412],  # Dirang
            [27.5012, 92.1012],  # Sela Pass (High Altitude Slide Choke)
            [27.5312, 91.9812],  # Jang
            [27.5861, 91.8653],  # Tawang
        ]
    },
    "NH-110 (Siliguri - Darjeeling)": {
        "associated_station": "Darjeeling",
        "highway_name": "NH-110 Hill Cart Road",
        "chokepoints": "Pagla Jhora, Tindharia, Kurseong Ghat",
        "coordinates": [
            [26.7271, 88.3953],  # Siliguri
            [26.8212, 88.3312],  # Sukna
            [26.8612, 88.3412],  # Tindharia
            [26.8812, 88.3112],  # Pagla Jhora (Active Sinking Water Choke)
            [26.9012, 88.2812],  # Kurseong
            [26.9812, 88.2712],  # Ghum
            [27.0360, 88.2627],  # Darjeeling
        ]
    },
    "NH-37 (Jiribam - Imphal)": {
        "associated_station": "Imphal",
        "highway_name": "NH-37 Manipur Alternate Lifeline",
        "chokepoints": "Makru Bridge, Nungba Ridge, Khongsang",
        "coordinates": [
            [24.8012, 93.1212],  # Jiribam Gateway
            [24.8512, 93.3112],  # Makru River Sinking Choke
            [24.7812, 93.5212],  # Nungba Ridge
            [24.8212, 93.7112],  # Khongsang
            [24.8112, 93.9312],  # Imphal Capital Valley
        ]
    }
}

# Incident Command Action Matrix SOP Protocol Definitions
SOP_LEVELS = [
    {
        "level": 1,
        "name": "LEVEL 1: ADVISORY",
        "range": (0.0, 0.35),
        "badge": "GREEN - NORMAL ROUTINE",
        "color": "#10b981",
        "border": "rgba(16, 185, 129, 0.4)",
        "action": "All arterial highways open. Routine telemetry acquisition active. No traffic restrictions."
    },
    {
        "level": 2,
        "name": "LEVEL 2: WATCH",
        "range": (0.35, 0.65),
        "badge": "YELLOW - BRO STANDBY",
        "color": "#f59e0b",
        "border": "rgba(245, 158, 11, 0.5)",
        "action": "BRO Directive: Pre-position JCBs and heavy earthmoving plant at active slide chokepoints. Halt night heavy freight."
    },
    {
        "level": 3,
        "name": "LEVEL 3: WARNING",
        "range": (0.65, 0.85),
        "badge": "ORANGE - TRAFFIC DIVERSION",
        "color": "#f97316",
        "border": "rgba(249, 115, 22, 0.6)",
        "action": "Traffic Police Directive: Divert civilian transit to alternative bypass routes. Mobilize SDRF search & rescue units."
    },
    {
        "level": 4,
        "name": "LEVEL 4: EVACUATION",
        "range": (0.85, 1.00),
        "badge": "RED - SEC 144 SHUTDOWN",
        "color": "#ef4444",
        "border": "rgba(239, 68, 68, 0.8)",
        "action": "MANDATORY HIGHWAY SHUTDOWN under Sec 144. Initiate immediate evacuation of downslope settlements to designated relief camps."
    }
]

CAP_LANGUAGES = {
    "English": "en-IN",
    "Hindi (हिन्दी)": "hi-IN",
    "Assamese (অসমীয়া)": "as-IN",
    "Bengali (বাংলা)": "bn-IN",
    "Mizo (Mizo ṭawng)": "lus-IN",
    "Nepali (नेपाली)": "ne-IN"
}

MULTILINGUAL_CAP_ALERTS = {
    "Extreme": {
        "urgency": "Immediate",
        "severity": "Extreme",
        "certainty": "Observed",
        "templates": {
            "English": {
                "headline": "CRITICAL RED ALERT: Imminent Slope Failure & Landslide Disaster",
                "instruction": "MANDATORY EVACUATION. Arterial highway closed under Sec 144. Move away from steep slopes immediately to designated relief shelters. Do not travel.",
            },
            "Hindi (हिन्दी)": {
                "headline": "अति गंभीर लाल चेतावनी: तत्काल भूस्खलन एवं ढलान विफलता का खतरा",
                "instruction": "अनिवार्य निकासी आदेश। धारा 144 के तहत राजमार्ग पूरी तरह बंद। तुरंत ढलानों से दूर नजदीकी राहत शिविरों में जाएं। यात्रा न करें।",
            },
            "Assamese (অসমীয়া)": {
                "headline": "জৰুৰী ৰঙা সতৰ্কবাৰ্তা: প্ৰচণ্ড ভূমিস্খলনৰ নিশ্চিত আশংকা",
                "instruction": "বাধ্যতামূলক খালীকৰণ নিৰ্দেশ। ১৪৪ ধাৰাৰ অধীনত ঘাইপথ বন্ধ কৰা হৈছে। তাৎক্ষণিকভাৱে ওখ আৰু বিপজ্জনক পাহাৰীয়া ঢালৰ পৰা নিৰাপদ আশ্ৰয় শিবিৰলৈ যাওক।",
            },
            "Bengali (বাংলা)": {
                "headline": "জরুরি লাল সতর্কতা: ভয়াবহ ভূমিধসের মারাত্মক আশঙ্কা",
                "instruction": "বাধ্যতামূলক উচ্ছেদ নির্দেশ। ১৪৪ ধারা জারি করে মহাসড়ক বন্ধ করা হয়েছে। অবিলম্বে ঝুঁকিপূর্ণ ঢালু এলাকা ছেড়ে নিকটবর্তী আশ্রয়কেন্দ্রে যান।",
            },
            "Mizo (Mizo ṭawng)": {
                "headline": "HRIATTIRNA HLAUHZAWNG SEN (RED ALERT): Leitlah rapthlak tak a thleng dawn",
                "instruction": "HMUN HAWISAN NGAI (EVACUATE). Dan 144 hnuaiah kawngpui khar a ni. Tlang pang hlauhawm atangin himna hmun/relief camp lamah insuan vat rawh u.",
            },
            "Nepali (नेपाली)": {
                "headline": "आपतकालीन रातो चेतावनी: तत्काल भीषण पहिरो जाने निश्चित सम्भावना",
                "instruction": "अनिवार्य खाली गर्ने आदेश। धारा १४४ अन्तर्गत मुख्य राजमार्ग बन्द गरिएको छ। तत्काल भीरपहराबाट टाढा सुरक्षित आश्रयस्थलमा जानुहोस्।",
            }
        }
    },
    "High": {
        "urgency": "Expected",
        "severity": "Severe",
        "certainty": "Likely",
        "templates": {
            "English": {
                "headline": "ORANGE WARNING: High Landslide Vulnerability Detected",
                "instruction": "BRO earthmovers stationed at slide points. Avoid night journeys on hill routes. Expect traffic diversions.",
            },
            "Hindi (हिन्दी)": {
                "headline": "नारंगी चेतावनी: तीव्र भूस्खलन और मलबा गिरने की उच्च संभावना",
                "instruction": "सीमा सड़क संगठन (BRO) भारी मशीनें तैनात। पहाड़ी मार्गों पर रात में यात्रा करने से बचें। वैकल्पिक मार्गों का प्रयोग करें।",
            },
            "Assamese (অসমীয়া)": {
                "headline": "কমলা সতৰ্কবাৰ্তা: ভূমিস্খলন আৰু শিল খহি পৰাৰ প্ৰবল সম্ভাৱনা",
                "instruction": "বিৰূপ বতৰৰ প্ৰতি লক্ষ্য ৰাখি নিশাৰ যাতায়াত স্থগিত ৰাখক। বিআৰঅ' (BRO) দলে কাম কৰি আছে। সাৱধান হওক।",
            },
            "Bengali (বাংলা)": {
                "headline": "কমলা সতর্কতা: ব্যাপক ভূমিধস ও শিলাপতনের সতর্কতা",
                "instruction": "পাহাড়ি পথে রাতের ভ্রমণ এড়িয়ে চলুন। বিআরও দুর্যোগ মোকাবেলায় প্রস্তুত। ট্রাফিক নির্দেশিকা অনুসরণ করুন।",
            },
            "Mizo (Mizo ṭawng)": {
                "headline": "VAWN-VUT HRIATTIRNA (ORANGE WARNING): Leitlah thleng thei a ni",
                "instruction": "Zan lamah kawngpui kal suh u. BRO ten kawng thenfai hna an thawk reng e. Fimkhur rawh u.",
            },
            "Nepali (नेपाली)": {
                "headline": "सुन्तला चेतावनी: भारी पहिरो खस्ने उच्च जोखिम",
                "instruction": "पहाडी सडकहरूमा रात्रिकालीन यात्रा नगर्नुहोस्। बीआरओ (BRO) टोली सतर्क अवस्थामा छ। ट्राफिक मार्गनिर्देशन पालना गर्नुहोस्।",
            }
        }
    },
    "Low": {
        "urgency": "Future",
        "severity": "Minor",
        "certainty": "Possible",
        "templates": {
            "English": {
                "headline": "GREEN ADVISORY: Normal Telemetry Monitoring Active",
                "instruction": "All hill corridors and highways open. Standard weather and seismic telemetry active.",
            },
            "Hindi (हिन्दी)": {
                "headline": "हरी सलाह: सामान्य निगरानी सक्रिय, स्थिति नियंत्रण में",
                "instruction": "सभी पहाड़ी गलियारे और राजमार्ग सामान्य रूप से खुले हैं। मानक मौसम टेलीमेट्री सक्रिय है।",
            },
            "Assamese (অসমীয়া)": {
                "headline": "সেউজ পৰামৰ্শ: সাধাৰণ স্থিতি আৰু নিয়মীয়া নিৰীক্ষণ কাৰ্যক্ষম",
                "instruction": "সকলো পাহাৰীয়া ঘাইপথ স্বাভাৱিকভাৱে খোলা আছে। বতৰ নিৰীক্ষণ কাৰ্যসূচী সক্ৰিয়।",
            },
            "Bengali (বাংলা)": {
                "headline": "সবুজ বার্তা: পরিস্থিতি স্বাভাবিক ও স্থিতিশীল",
                "instruction": "সমস্ত পাহাড়ি মহাসড়ক স্বাভাবিকভাবে উন্মুক্ত রয়েছে। রুটিন নজরদারি চালু আছে।",
            },
            "Mizo (Mizo ṭawng)": {
                "headline": "HRIATTIRNA HRENG (GREEN ADVISORY): Boruak a pangngai e",
                "instruction": "Kawngpui zawng zawng a tluang e. Telemetry hmanga vil chhunzawm zel a ni.",
            },
            "Nepali (नेपाली)": {
                "headline": "हरियो सल्लाह: सामान्य अनुगमन सक्रिय, अवस्था स्थिर",
                "instruction": "सबै पहाडी सडक तथा राजमार्गहरू सुचारु छन्। नियमित मौसम टेलिमेट्री सक्रिय छ।",
            }
        }
    }
}