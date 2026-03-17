"""
Event Classification Module
Classifies detected incidents into different types
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from datetime import datetime

from object_detector import Detection
from tracker import Track
from scene_analyzer import CrowdRegion, Anomaly
from config import EVENT_CLASSIFIER_CONFIG, LOGGING_CONFIG

# Setup logging
logging.basicConfig(level=LOGGING_CONFIG['log_level'])
logger = logging.getLogger(__name__)


@dataclass
class Incident:
    """Represents a detected incident"""
    incident_id: int
    incident_type: str
    confidence: float
    location: Tuple[int, int]
    description: str
    timestamp: float
    frame_count: int
    evidence: Dict = field(default_factory=dict)  # Supporting evidence
    duration: int = 0  # frames
    
    def __str__(self):
        return (f"Incident {self.incident_id}: {self.incident_type} "
                f"(conf={self.confidence:.2f}) at {self.location}")


class EventClassifier:
    """
    Classifies scenes and detects incidents
    """
    
    def __init__(self):
        """Initialize event classifier"""
        self.incidents: Dict[int, Incident] = {}
        self.next_incident_id = 1
        self.frame_count = 0
        self.recent_incidents = defaultdict(list)
        
        logger.info("Event Classifier initialized")
    
    def classify_incident(self,
                         frame_h: int, frame_w: int,
                         detections: List[Detection],
                         tracks: List[Track],
                         crowd_regions: List[CrowdRegion],
                         anomalies: List[Anomaly],
                         unauthorized_entries: List[Tuple[int, int, str]],
                         vehicle_stops: List[Tuple[int, str]]) -> List[Incident]:
        """
        Classify current scene and generate incidents
        
        Args:
            frame_h, frame_w: Frame dimensions
            detections: Current detections
            tracks: Current tracks
            crowd_regions: Crowd density regions
            anomalies: Detected anomalies
            unauthorized_entries: Unauthorized zone entries
            vehicle_stops: Suspicious vehicle stops
            
        Returns:
            List of new incidents detected in this frame
        """
        self.frame_count += 1
        new_incidents = []
        
        # Analyze crowd gathering
        crowd_incidents = self._analyze_crowd_gathering(
            detections, crowd_regions
        )
        new_incidents.extend(crowd_incidents)
        
        # Analyze accidents/falls
        fall_incidents = self._analyze_falls(detections, tracks, anomalies)
        new_incidents.extend(fall_incidents)
        
        # Analyze unauthorized entry
        entry_incidents = self._analyze_unauthorized_entry(unauthorized_entries)
        new_incidents.extend(entry_incidents)
        
        # Analyze suspicious vehicles
        vehicle_incidents = self._analyze_suspicious_vehicles(vehicle_stops)
        new_incidents.extend(vehicle_incidents)
        
        # Analyze abandoned objects
        object_incidents = self._analyze_abandoned_objects(
            detections, tracks, anomalies
        )
        new_incidents.extend(object_incidents)
        
        # Register and track incidents
        for incident in new_incidents:
            incident.incident_id = self.next_incident_id
            incident.timestamp = datetime.now().timestamp()
            incident.frame_count = self.frame_count
            
            self.incidents[self.next_incident_id] = incident
            self.recent_incidents[incident.incident_type].append(incident)
            
            self.next_incident_id += 1
        
        # Update incident durations
        for incident in self.incidents.values():
            incident.duration = self.frame_count - incident.frame_count
        
        # Remove old incidents
        self._cleanup_old_incidents()
        
        return new_incidents
    
    def _analyze_crowd_gathering(self,
                                 detections: List[Detection],
                                 crowd_regions: List[CrowdRegion]) -> List[Incident]:
        """Detect sudden crowd gathering"""
        incidents = []
        
        crowded_regions = [r for r in crowd_regions if r.is_crowd]
        
        if not crowded_regions:
            return incidents
        
        people = [d for d in detections if d.class_name == 'person']
        
        for region in crowded_regions:
            # High confidence if high density
            confidence = min(1.0, region.density / 2.0)
            
            incident = Incident(
                incident_id=0,  # Will be assigned later
                incident_type='crowd_gathering',
                confidence=confidence,
                location=region.center,
                description=f"Sudden crowd gathering: {region.count} people "
                           f"(density: {region.density:.2f})",
                timestamp=0,
                frame_count=0,
                evidence={
                    'people_count': region.count,
                    'density': region.density,
                    'region': region,
                }
            )
            
            incidents.append(incident)
        
        return incidents
    
    def _analyze_falls(self,
                      detections: List[Detection],
                      tracks: List[Track],
                      anomalies: List[Anomaly]) -> List[Incident]:
        """Detect falls and accidents"""
        incidents = []
        
        # Sudden movement anomalies might indicate falls
        sudden_movement_anomalies = [a for a in anomalies 
                                    if a.anomaly_type == 'sudden_movement']
        
        for anomaly in sudden_movement_anomalies:
            if anomaly.confidence > 0.7:  # High confidence
                incident = Incident(
                    incident_id=0,
                    incident_type='road_accident',
                    confidence=anomaly.confidence,
                    location=(0, 0),  # Will be updated below
                    description=anomaly.description,
                    timestamp=0,
                    frame_count=0,
                    evidence={'anomaly': anomaly}
                )
                
                # Get location from track
                for track in tracks:
                    if track.track_id == anomaly.track_id:
                        pos = track.get_current_position()
                        if pos:
                            incident.location = pos
                        break
                
                incidents.append(incident)
        
        return incidents
    
    def _analyze_unauthorized_entry(self,
                                   entries: List[Tuple[int, int, str]]) -> List[Incident]:
        """Detect unauthorized zone entry"""
        incidents = []
        
        for x, y, zone_name in entries:
            incident = Incident(
                incident_id=0,
                incident_type='unauthorized_entry',
                confidence=0.9,
                location=(x, y),
                description=f"Unauthorized entry into {zone_name}",
                timestamp=0,
                frame_count=0,
                evidence={'zone': zone_name}
            )
            
            incidents.append(incident)
        
        return incidents
    
    def _analyze_suspicious_vehicles(self,
                                    vehicles: List[Tuple[int, str]]) -> List[Incident]:
        """Detect suspicious vehicle behavior"""
        incidents = []
        
        for track_id, description in vehicles:
            incident = Incident(
                incident_id=0,
                incident_type='suspicious_vehicle',
                confidence=0.8,
                location=(0, 0),
                description=description,
                timestamp=0,
                frame_count=0,
                evidence={'track_id': track_id}
            )
            
            incidents.append(incident)
        
        return incidents
    
    def _analyze_abandoned_objects(self,
                                  detections: List[Detection],
                                  tracks: List[Track],
                                  anomalies: List[Anomaly]) -> List[Incident]:
        """Detect abandoned objects"""
        incidents = []
        
        # Abandoned object anomalies
        abandoned_anomalies = [a for a in anomalies 
                              if a.anomaly_type == 'abandoned_object']
        
        for anomaly in abandoned_anomalies:
            incident = Incident(
                incident_id=0,
                incident_type='abandoned_object',
                confidence=anomaly.confidence,
                location=(0, 0),
                description=anomaly.description,
                timestamp=0,
                frame_count=0,
                evidence={'anomaly': anomaly}
            )
            
            # Get location from track
            for track in tracks:
                if track.track_id == anomaly.track_id:
                    pos = track.get_current_position()
                    if pos:
                        incident.location = pos
                    break
            
            incidents.append(incident)
        
        return incidents
    
    def _cleanup_old_incidents(self):
        """Remove incidents that are too old"""
        max_duration = EVENT_CLASSIFIER_CONFIG['max_incident_duration']
        
        old_incident_ids = []
        for incident_id, incident in self.incidents.items():
            if incident.duration > max_duration:
                old_incident_ids.append(incident_id)
        
        for incident_id in old_incident_ids:
            del self.incidents[incident_id]
    
    def get_active_incidents(self) -> List[Incident]:
        """Get all active incidents"""
        return list(self.incidents.values())
    
    def get_incidents_by_type(self, incident_type: str) -> List[Incident]:
        """Get incidents of specific type"""
        return [i for i in self.incidents.values() 
                if i.incident_type == incident_type]
    
    def get_critical_incidents(self, 
                              min_confidence: float = None) -> List[Incident]:
        """Get high-confidence incidents"""
        min_conf = min_confidence or EVENT_CLASSIFIER_CONFIG['confidence_threshold']
        return [i for i in self.incidents.values() 
                if i.confidence >= min_conf]
    
    def get_incident_summary(self) -> Dict:
        """Get summary of incidents"""
        summary = {
            'total_incidents': len(self.incidents),
            'by_type': defaultdict(int),
            'by_confidence': {'high': 0, 'medium': 0, 'low': 0},
        }
        
        for incident in self.incidents.values():
            summary['by_type'][incident.incident_type] += 1
            
            if incident.confidence >= 0.8:
                summary['by_confidence']['high'] += 1
            elif incident.confidence >= 0.6:
                summary['by_confidence']['medium'] += 1
            else:
                summary['by_confidence']['low'] += 1
        
        return summary
    
    def reset(self):
        """Reset classifier state"""
        self.incidents.clear()
        self.recent_incidents.clear()
        self.frame_count = 0
        self.next_incident_id = 1


def draw_incidents(frame: np.ndarray,
                  incidents: List[Incident]) -> np.ndarray:
    """
    Draw incidents on frame
    
    Args:
        frame: Input frame
        incidents: List of incidents
        
    Returns:
        Frame with drawn incidents
    """
    import cv2
    
    frame_copy = frame.copy()
    
    for incident in incidents:
        x, y = incident.location
        
        # Color based on type
        color_map = {
            'crowd_gathering': (0, 255, 255),      # Yellow
            'road_accident': (0, 0, 255),          # Red
            'unauthorized_entry': (0, 165, 255),   # Orange
            'suspicious_vehicle': (255, 0, 0),     # Blue
            'abandoned_object': (128, 0, 128),     # Purple
        }
        
        color = color_map.get(incident.incident_type, (128, 128, 128))
        
        # Draw incident marker
        cv2.circle(frame_copy, (x, y), 20, color, 3)
        cv2.circle(frame_copy, (x, y), 30, color, 1)
        
        # Draw label
        label = f"[{incident.incident_id}] {incident.incident_type.replace('_', ' ').upper()}"
        font_scale = 0.5
        thickness = 1
        
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                    font_scale, thickness)[0]
        
        text_x = max(0, x - text_size[0] // 2)
        text_y = max(20, y - 40)
        
        cv2.rectangle(frame_copy, 
                     (text_x - 2, text_y - text_size[1] - 2),
                     (text_x + text_size[0] + 2, text_y + 2),
                     color, -1)
        
        cv2.putText(frame_copy, label, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                   (255, 255, 255), thickness)
        
        # Draw confidence
        conf_text = f"Conf: {incident.confidence:.2f}"
        cv2.putText(frame_copy, conf_text, (x - 30, y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return frame_copy


# Example usage
if __name__ == "__main__":
    import cv2
    
    classifier = EventClassifier()
    
    # Create fake frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Simulate incident
    detections = []
    tracks = []
    crowd_regions = [
        CrowdRegion(x=100, y=100, density=1.5, count=10, 
                   center=(125, 125), is_crowd=True)
    ]
    anomalies = []
    unauthorized_entries = []
    vehicle_stops = []
    
    incidents = classifier.classify_incident(
        480, 640, detections, tracks, crowd_regions, 
        anomalies, unauthorized_entries, vehicle_stops
    )
    
    print(f"Detected incidents: {len(incidents)}")
    for incident in incidents:
        print(f"  {incident}")
    
    frame = draw_incidents(frame, incidents)
    
    cv2.imshow('Incidents', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
