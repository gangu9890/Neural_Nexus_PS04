# Urban Safety Detection Pipeline

AI-powered system for detecting safety incidents in urban environments using computer vision, multi-object tracking, and machine learning.

## System Architecture

```
Video Input
    ↓
[Frame Extraction] - Extract frames from video/webcam
    ↓
[YOLOv8 Object Detection] - Detect people, vehicles, objects
    ↓
[Multi-Object Tracking] - Track movement patterns (ByteTrack)
    ↓
[Scene Analysis] - Detect crowds, anomalies, unauthorized entry
    ↓
[Event Classification] - Classify incidents and generate alerts
    ↓
[Drone Dispatch] - Autonomous response system
```

## Incident Types Detected

1. **Crowd Gathering** - Sudden crowd formation in low-density areas
2. **Road Accident** - Falls, collisions, sudden movements
3. **Unauthorized Entry** - Entry into restricted zones
4. **Suspicious Vehicle** - Stationary vehicles in unusual locations
5. **Abandoned Objects** - Unattended bags, luggage
6. **Traffic Violations** - Rule violations (optional)

## Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU (optional, for faster inference)
- 8GB+ RAM
- 2GB+ disk space for models

### Setup

1. **Clone/Create project directory**
```bash
mkdir safety-detection
cd safety-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download YOLOv8 models** (automatic on first run)
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

## Configuration

Edit `config.py` to customize:

### Video Settings
```python
VIDEO_CONFIG = {
    'input_source': 0,  # 0 for webcam, or path to video file
    'frame_width': 640,
    'frame_height': 480,
}
```

### Detection Settings
```python
YOLO_CONFIG = {
    'model_name': 'yolov8m.pt',  # nano, small, medium, large
    'confidence_threshold': 0.5,
    'device': 'cuda',  # or 'cpu'
}
```

### Tracking Settings
```python
TRACKING_CONFIG = {
    'tracker_type': 'bytetrack',
    'max_age': 30,  # frames
    'min_hits': 3,
}
```

### Scene Analysis
```python
SCENE_ANALYSIS_CONFIG = {
    'min_crowd_size': 5,
    'crowd_density_threshold': 0.3,
    'sudden_movement_threshold': 50,  # pixels
    'abandoned_frames_threshold': 300,  # ~10 seconds
}
```

### Restricted Zones
```python
RESTRICTED_ZONES = [
    {
        'name': 'Zone A',
        'center': (300, 200),
        'radius': 100,
    },
]
```

## Usage

### Basic Usage (Webcam)
```bash
python main.py
```

### With Video File
```bash
python main.py --video path/to/video.mp4
```

### Save Output Video
```bash
python main.py --video input.mp4 --output output.mp4
```

### Limit Processing
```bash
python main.py --max-frames 1000
```

### Command Line Options
```bash
python main.py --help

Options:
  --video VIDEO           Video file path or camera index (default: 0)
  --output OUTPUT         Output video file path
  --max-frames MAX_FRAMES Maximum frames to process
  --no-fps               Don't display FPS counter
```

## Module Details

### 1. Frame Extractor (`frame_extractor.py`)
- Reads frames from video files or webcam
- Handles video preprocessing
- Supports frame skipping for real-time performance

**Key Classes:**
- `FrameExtractor` - Main frame extraction class
- `VideoWriter` - Writes processed frames to video file

### 2. Object Detector (`object_detector.py`)
- YOLOv8-based detection
- Detects people, vehicles, and objects
- Confidence filtering and NMS

**Key Classes:**
- `Detection` - Represents single detection
- `ObjectDetector` - YOLOv8 wrapper

**Methods:**
```python
detector.detect(frame)  # Run detection
detector.get_person_detections(detections)
detector.get_vehicle_detections(detections)
detector.draw_detections(frame, detections)
```

### 3. Multi-Object Tracker (`tracker.py`)
- ByteTrack-based tracking
- Tracks movement patterns
- Calculates velocity and direction

**Key Classes:**
- `Track` - Represents tracked object
- `SimpleTracker` - Basic centroid-based tracker
- `ByteTrackWrapper` - Confidence-weighted tracking

**Methods:**
```python
tracker.update(detections)  # Update with new detections
tracker.get_all_tracks()
tracker.reset()
```

### 4. Scene Analyzer (`scene_analyzer.py`)
- Crowd density analysis
- Anomaly detection
- Restricted zone enforcement

**Key Classes:**
- `CrowdRegion` - Crowd detection region
- `Anomaly` - Detected anomaly
- `SceneAnalyzer` - Main analyzer

**Methods:**
```python
analyzer.analyze_crowd_density(frame, detections)
analyzer.detect_anomalies(tracks)
analyzer.detect_unauthorized_entry(frame, detections, zones)
analyzer.detect_vehicle_stops(tracks)
```

### 5. Event Classifier (`event_classifier.py`)
- Classifies incidents
- Generates incident alerts
- Tracks incident duration

**Key Classes:**
- `Incident` - Represents detected incident
- `EventClassifier` - Classification engine

**Methods:**
```python
classifier.classify_incident(...)
classifier.get_critical_incidents(min_confidence=0.8)
classifier.get_incident_summary()
```

## Module Integration Flow

```python
# Example integration
pipeline = SafetyDetectionPipeline(video_source=0)

# Get frame
success, frame = extractor.get_frame()

# Detect objects
detections = detector.detect(frame)

# Track objects
tracks = tracker.update(detections)

# Analyze scene
crowds = analyzer.analyze_crowd_density(frame, detections)
anomalies = analyzer.detect_anomalies(tracks)

# Classify incidents
incidents = classifier.classify_incident(
    frame_h, frame_w, detections, tracks, 
    crowds, anomalies, entries, stops
)

# Visualize
frame = draw_incidents(frame, incidents)
cv2.imshow('Result', frame)
```

## Performance

### Typical FPS (GPU)
- **YOLOv8n** (nano): 50-60 FPS
- **YOLOv8s** (small): 40-50 FPS
- **YOLOv8m** (medium): 25-35 FPS
- **YOLOv8l** (large): 15-25 FPS

### Memory Usage
- YOLOv8m: ~3-4 GB GPU
- Tracker: ~500 MB
- Total: ~4-5 GB

## Output

### Console Output
```
Frame 100: 5 detection(s), 3 track(s), 1 incident(s) detected
  Incident 1: crowd_gathering (conf=0.92) at (320, 240)
```

### Video Output
- Bounding boxes with class labels
- Track IDs and trajectories
- Incident markers and descriptions
- Crowd density heatmaps
- Anomaly indicators

### Statistics File
`pipeline_stats.json`:
```json
{
  "frames_processed": 1000,
  "total_detections": 5432,
  "total_tracks": 234,
  "total_incidents": 42,
  "incidents_by_type": {
    "crowd_gathering": 15,
    "road_accident": 8,
    "unauthorized_entry": 12,
    "suspicious_vehicle": 7
  }
}
```

## Customization

### Add Custom Detection Class
```python
# In config.py
DETECTION_CLASSES = {
    0: 'person',
    1: 'bicycle',
    # ... add more classes
}
```

### Modify Incident Thresholds
```python
# In config.py
SCENE_ANALYSIS_CONFIG = {
    'min_crowd_size': 3,  # Lower for more sensitivity
    'sudden_movement_threshold': 30,  # Lower threshold
}
```

### Custom Drawing Style
```python
# In config.py
VISUALIZATION_CONFIG = {
    'colors': {
        'person': (0, 255, 0),
        'vehicle': (0, 0, 255),
        # ... customize colors
    }
}
```

## Troubleshooting

### GPU Not Detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU
# In config.py: YOLO_CONFIG['device'] = 'cpu'
```

### Slow FPS
- Use smaller model: `'yolov8n.pt'` or `'yolov8s.pt'`
- Increase frame skip: `VIDEO_CONFIG['skip_frames'] = 2`
- Reduce frame size: `VIDEO_CONFIG['frame_width'] = 416`

### No Detections
- Lower confidence threshold: `YOLO_CONFIG['confidence_threshold'] = 0.3`
- Check video source quality
- Ensure good lighting

### High False Positives
- Increase confidence threshold: `YOLO_CONFIG['confidence_threshold'] = 0.7`
- Adjust anomaly thresholds
- Tune crowd density parameters

## Advanced Usage

### Real-Time Monitoring Dashboard
```python
from main import SafetyDetectionPipeline

pipeline = SafetyDetectionPipeline(video_source='rtsp://camera_url')
pipeline.run()
```

### Multi-Camera Setup
```bash
# Run separate instances for each camera
python main.py --video camera_1.mp4 --output output_1.mp4 &
python main.py --video camera_2.mp4 --output output_2.mp4 &
```

### Integration with Drone System
```python
from main import SafetyDetectionPipeline
from drone_controller import dispatch_drone

pipeline = SafetyDetectionPipeline()
while True:
    incidents = pipeline.frame_extractor.get_frame()
    
    for incident in incidents:
        if incident.confidence > 0.8:
            dispatch_drone(incident.location)
```

## License

Project for INSPIRON 5.0 Hackathon

## Support

For issues and questions:
1. Check configuration in `config.py`
2. Review module documentation in respective files
3. Check console output for error messages
4. Enable debug logging: `LOGGING_CONFIG['log_level'] = 'DEBUG'`

## Future Enhancements

- [ ] Real-time web dashboard
- [ ] Multi-GPU support
- [ ] Custom model fine-tuning
- [ ] Database logging
- [ ] REST API for integration
- [ ] Autonomous drone dispatch
- [ ] Heat map generation
- [ ] Predictive analytics

---

**Created for INSPIRON 5.0 Hackathon**
Computer Society of India - COEP Tech Student Chapter
