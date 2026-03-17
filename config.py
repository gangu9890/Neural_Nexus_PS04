"""
Configuration file for Urban Safety Detection System
"""

import os
from typing import Dict, List

# ======================== VIDEO CONFIGURATION ========================
VIDEO_CONFIG = {
    'input_source': 0,  # 0 for webcam, or path to video file
    'fps': 30,
    'frame_width': 640,
    'frame_height': 480,
    'skip_frames': 1,  # Process every nth frame (1 = process all)
}

# ======================== OBJECT DETECTION CONFIG ========================
YOLO_CONFIG = {
    'model_name': 'yolov8m.pt',  # nano, small, medium, large, xlarge
    'confidence_threshold': 0.5,
    'iou_threshold': 0.45,
    'device': 'cuda',  # 'cuda' or 'cpu'
    'max_det': 300,
}

# COCO classes we care about
DETECTION_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
}

# ======================== TRACKING CONFIG ========================
TRACKING_CONFIG = {
    'tracker_type': 'bytetrack',  # 'bytetrack' or 'deepsort'
    'max_age': 30,  # Maximum frames to keep track alive without detection
    'min_hits': 3,  # Minimum detections before confirming track
    'iou_threshold': 0.3,
}

# ======================== SCENE ANALYSIS CONFIG ========================
SCENE_ANALYSIS_CONFIG = {
    # Crowd Detection
    'crowd_detection_enabled': True,
    'min_crowd_size': 5,  # Minimum people for crowd
    'crowd_density_threshold': 0.3,  # People per pixel area
    'crowd_region_size': 50,  # Size of grid cell for density calculation
    
    # Anomaly Detection
    'anomaly_detection_enabled': True,
    'sudden_movement_threshold': 50,  # Pixels per frame
    'stationary_threshold': 5,  # Frames to consider stationary
    'velocity_outlier_threshold': 2.0,  # Standard deviations
    
    # Abandoned Objects
    'abandoned_object_enabled': True,
    'abandoned_frames_threshold': 300,  # ~10 seconds at 30fps
    'min_object_area': 100,  # Min pixels
}

# ======================== EVENT CLASSIFICATION CONFIG ========================
EVENT_CLASSIFIER_CONFIG = {
    'incident_types': [
        'crowd_gathering',      # Sudden crowd in low-density area
        'road_accident',        # Fallen person or vehicle collision
        'unauthorized_entry',   # Entry into restricted zone
        'suspicious_vehicle',   # Stopped vehicle in suspicious location
        'abandoned_object',     # Object left unattended
        'traffic_violation',    # Vehicle rule violation
        'normal_activity',      # No incident
    ],
    
    'confidence_threshold': 0.6,
    'max_incident_duration': 600,  # Max seconds to track one incident
}

# ======================== GEOLOCATION CONFIG ========================
GEOLOCATION_CONFIG = {
    'reference_latitude': 19.0760,  # Pune coordinates (example)
    'reference_longitude': 72.8777,
    'pixels_per_meter': 0.05,  # Calibration factor (adjust based on camera)
}

# ======================== DRONE DISPATCH CONFIG ========================
DRONE_DISPATCH_CONFIG = {
    'drone_speed_kmh': 50,  # Km/h
    'max_drones': 5,
    'dispatch_cooldown': 30,  # Seconds between dispatches
    'alert_priority_levels': {
        'critical': 1,  # Road accident, fall
        'high': 2,      # Crowd gathering
        'medium': 3,    # Abandoned object
        'low': 4,       # Traffic violation
    }
}

# ======================== VISUALIZATION CONFIG ========================
VISUALIZATION_CONFIG = {
    'show_detections': True,
    'show_tracks': True,
    'show_crowd_heatmap': True,
    'show_anomalies': True,
    'font_scale': 0.6,
    'line_thickness': 2,
    
    # Colors (BGR format)
    'colors': {
        'person': (0, 255, 0),      # Green
        'vehicle': (0, 0, 255),     # Red
        'object': (255, 0, 0),      # Blue
        'crowd': (0, 255, 255),     # Yellow
        'anomaly': (255, 0, 255),   # Magenta
        'alert': (0, 165, 255),     # Orange
    }
}

# ======================== LOGGING CONFIG ========================
LOGGING_CONFIG = {
    'log_dir': './logs',
    'log_level': 'INFO',
    'save_detections': True,
    'save_tracks': True,
    'save_incidents': True,
}

# ======================== PERFORMANCE CONFIG ========================
PERFORMANCE_CONFIG = {
    'enable_gpu': True,
    'batch_size': 1,
    'num_workers': 2,
    'inference_mode': True,  # torch.inference_mode for faster inference
}

# ======================== RESTRICTED ZONES (for geofencing) ========================
RESTRICTED_ZONES = [
    {
        'name': 'Zone A',
        'center': (300, 200),
        'radius': 100,
    },
    {
        'name': 'Zone B',
        'center': (500, 400),
        'radius': 150,
    },
]

# Create log directory if it doesn't exist
os.makedirs(LOGGING_CONFIG['log_dir'], exist_ok=True)
