"""
Scene Analysis Module
Analyzes scene for crowd density, anomalies, and suspicious activities
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

from object_detector import Detection
from tracker import Track
from config import SCENE_ANALYSIS_CONFIG, LOGGING_CONFIG

# Setup logging
logging.basicConfig(level=LOGGING_CONFIG['log_level'])
logger = logging.getLogger(__name__)


@dataclass
class CrowdRegion:
    """Represents a crowd region"""
    x: int
    y: int
    density: float
    count: int
    center: Tuple[int, int]
    is_crowd: bool


@dataclass
class Anomaly:
    """Represents detected anomaly"""
    anomaly_type: str  # 'sudden_movement', 'stationary', 'high_velocity'
    track_id: int
    confidence: float
    description: str


class SceneAnalyzer:
    """
    Analyzes scene for various incidents
    """
    
    def __init__(self):
        """Initialize scene analyzer"""
        self.crowd_history = defaultdict(list)
        self.velocity_history = defaultdict(list)
        self.stationarity_history = defaultdict(int)
        
        logger.info("Scene Analyzer initialized")
    
    def analyze_crowd_density(self, frame: np.ndarray,
                             detections: List[Detection]) -> List[CrowdRegion]:
        """
        Analyze crowd density in different regions
        
        Args:
            frame: Input frame
            detections: List of detections (mainly people)
            
        Returns:
            List of CrowdRegion objects
        """
        frame_h, frame_w = frame.shape[:2]
        
        # Get person detections only
        people = [d for d in detections if d.class_name == 'person']
        
        if len(people) == 0:
            return []
        
        # Divide frame into grid
        region_size = SCENE_ANALYSIS_CONFIG['crowd_region_size']
        regions = []
        
        # Create grid
        grid_h = (frame_h + region_size - 1) // region_size
        grid_w = (frame_w + region_size - 1) // region_size
        
        for gy in range(grid_h):
            for gx in range(grid_w):
                # Region bounds
                x1 = gx * region_size
                y1 = gy * region_size
                x2 = min((gx + 1) * region_size, frame_w)
                y2 = min((gy + 1) * region_size, frame_h)
                
                region_area = (x2 - x1) * (y2 - y1)
                
                # Count people in region
                people_count = 0
                for person in people:
                    if (person.center[0] >= x1 and person.center[0] < x2 and
                        person.center[1] >= y1 and person.center[1] < y2):
                        people_count += 1
                
                # Calculate density
                density = people_count / (region_area / 10000.0)  # per 10k pixels
                
                # Check if crowd
                is_crowd = (people_count >= SCENE_ANALYSIS_CONFIG['min_crowd_size'] and
                           density >= SCENE_ANALYSIS_CONFIG['crowd_density_threshold'])
                
                region = CrowdRegion(
                    x=x1,
                    y=y1,
                    density=density,
                    count=people_count,
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                    is_crowd=is_crowd,
                )
                
                regions.append(region)
        
        return regions
    
    def detect_anomalies(self, tracks: List[Track]) -> List[Anomaly]:
        """
        Detect anomalies in track behavior
        
        Args:
            tracks: List of tracks
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        for track in tracks:
            # Detect sudden movement
            if track.max_velocity > SCENE_ANALYSIS_CONFIG['sudden_movement_threshold']:
                anomaly = Anomaly(
                    anomaly_type='sudden_movement',
                    track_id=track.track_id,
                    confidence=min(1.0, track.max_velocity / 100.0),
                    description=f"{track.class_name} moving suddenly at "
                                f"{track.max_velocity:.1f} px/frame",
                )
                anomalies.append(anomaly)
            
            # Detect stationary objects
            if (len(track.positions) > SCENE_ANALYSIS_CONFIG['stationary_threshold'] and
                track.max_velocity < 5):  # Stationary threshold
                self.stationarity_history[track.track_id] += 1
                
                if self.stationarity_history[track.track_id] > 30:  # ~1 second
                    if track.class_name in ['backpack', 'handbag', 'suitcase']:
                        anomaly = Anomaly(
                            anomaly_type='abandoned_object',
                            track_id=track.track_id,
                            confidence=0.8,
                            description=f"Abandoned {track.class_name}",
                        )
                        anomalies.append(anomaly)
            else:
                self.stationarity_history[track.track_id] = 0
            
            # Detect high velocity (unusual for some objects)
            if track.class_name == 'person':
                velocity = track.get_velocity()
                self.velocity_history[track.track_id].append(velocity)
                
                if len(self.velocity_history[track.track_id]) > 10:
                    velocities = self.velocity_history[track.track_id][-10:]
                    mean_vel = np.mean(velocities)
                    std_vel = np.std(velocities)
                    
                    if velocity > mean_vel + 2 * std_vel:
                        anomaly = Anomaly(
                            anomaly_type='high_velocity',
                            track_id=track.track_id,
                            confidence=0.7,
                            description=f"Person moving at unusual speed "
                                       f"{velocity:.2f}",
                        )
                        anomalies.append(anomaly)
        
        return anomalies
    
    def detect_unauthorized_entry(self, frame: np.ndarray,
                                  detections: List[Detection],
                                  restricted_zones: List[Dict]) -> List[Tuple[int, int, str]]:
        """
        Detect entry into restricted zones
        
        Args:
            frame: Input frame
            detections: List of detections
            restricted_zones: List of restricted zone definitions
            
        Returns:
            List of (x, y, zone_name) violations
        """
        violations = []
        
        if not restricted_zones:
            return violations
        
        for detection in detections:
            for zone in restricted_zones:
                if self._point_in_zone(detection.center, zone):
                    violations.append((detection.center[0], detection.center[1], 
                                     zone.get('name', 'Unknown Zone')))
        
        return violations
    
    @staticmethod
    def _point_in_zone(point: Tuple[int, int], zone: Dict) -> bool:
        """Check if point is in zone"""
        center = zone.get('center')
        radius = zone.get('radius', 50)
        
        if center is None:
            return False
        
        dist = np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
        return dist <= radius
    
    def detect_vehicle_stops(self, tracks: List[Track],
                           suspicious_threshold: int = 30) -> List[Tuple[int, str]]:
        """
        Detect suspicious vehicle stops
        
        Args:
            tracks: List of tracks
            suspicious_threshold: Frames before considering suspicious
            
        Returns:
            List of (track_id, description) suspicious vehicles
        """
        suspicious_vehicles = []
        
        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
        
        for track in tracks:
            if track.class_name not in vehicle_classes:
                continue
            
            # Check if vehicle is stationary
            if (len(track.positions) > suspicious_threshold and
                track.max_velocity < 3):
                
                suspicious_vehicles.append((
                    track.track_id,
                    f"Suspicious {track.class_name} stop at "
                    f"{track.get_current_position()}"
                ))
        
        return suspicious_vehicles
    
    def get_scene_stats(self, frame: np.ndarray,
                       detections: List[Detection],
                       tracks: List[Track],
                       crowd_regions: List[CrowdRegion]) -> Dict:
        """
        Get comprehensive scene statistics
        
        Args:
            frame: Input frame
            detections: List of detections
            tracks: List of tracks
            crowd_regions: List of crowd regions
            
        Returns:
            Dictionary of scene statistics
        """
        people = [d for d in detections if d.class_name == 'person']
        vehicles = [d for d in detections 
                   if d.class_name in ['car', 'truck', 'bus']]
        objects = [d for d in detections 
                  if d.class_name in ['backpack', 'handbag', 'suitcase']]
        
        crowded_regions = [r for r in crowd_regions if r.is_crowd]
        
        return {
            'total_people': len(people),
            'total_vehicles': len(vehicles),
            'total_objects': len(objects),
            'total_tracks': len(tracks),
            'crowded_regions': len(crowded_regions),
            'max_crowd_density': max([r.density for r in crowd_regions]) 
                                if crowd_regions else 0,
            'frame_shape': frame.shape,
        }
    
    def reset(self):
        """Reset analyzer state"""
        self.crowd_history.clear()
        self.velocity_history.clear()
        self.stationarity_history.clear()


def draw_crowd_regions(frame: np.ndarray,
                      crowd_regions: List[CrowdRegion]) -> np.ndarray:
    """
    Draw crowd regions on frame
    
    Args:
        frame: Input frame
        crowd_regions: List of crowd regions
        
    Returns:
        Frame with drawn regions
    """
    frame_copy = frame.copy()
    
    for region in crowd_regions:
        # Draw region
        color = (0, 0, 255) if region.is_crowd else (0, 255, 0)
        thickness = 2 if region.is_crowd else 1
        
        cv2.rectangle(frame_copy, (region.x, region.y),
                     (region.x + SCENE_ANALYSIS_CONFIG['crowd_region_size'],
                      region.y + SCENE_ANALYSIS_CONFIG['crowd_region_size']),
                     color, thickness)
        
        # Draw density info
        if region.count > 0:
            text = f"{region.count}p"
            cv2.putText(frame_copy, text, (region.x + 5, region.y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame_copy


def draw_anomalies(frame: np.ndarray,
                  anomalies: List[Anomaly],
                  tracks: List[Track]) -> np.ndarray:
    """
    Draw anomalies on frame
    
    Args:
        frame: Input frame
        anomalies: List of anomalies
        tracks: List of tracks (for getting positions)
        
    Returns:
        Frame with drawn anomalies
    """
    frame_copy = frame.copy()
    
    # Create track position lookup
    track_positions = {track.track_id: track.get_current_position() 
                       for track in tracks}
    
    for anomaly in anomalies:
        pos = track_positions.get(anomaly.track_id)
        if pos is None:
            continue
        
        # Draw anomaly indicator
        color = (0, 0, 255)  # Red
        radius = 15
        thickness = 2
        
        cv2.circle(frame_copy, pos, radius, color, thickness)
        cv2.circle(frame_copy, pos, radius + 5, color, 1)
        
        # Draw text
        text = anomaly.anomaly_type.replace('_', ' ').upper()
        cv2.putText(frame_copy, text, (pos[0] - 40, pos[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame_copy


def draw_restricted_zones(frame: np.ndarray,
                         restricted_zones: List[Dict]) -> np.ndarray:
    """
    Draw restricted zones on frame
    
    Args:
        frame: Input frame
        restricted_zones: List of zone definitions
        
    Returns:
        Frame with drawn zones
    """
    frame_copy = frame.copy()
    
    for zone in restricted_zones:
        center = zone.get('center')
        radius = zone.get('radius', 50)
        name = zone.get('name', 'Zone')
        
        if center:
            # Draw zone circle
            cv2.circle(frame_copy, center, radius, (0, 165, 255), 2)
            
            # Draw zone name
            cv2.putText(frame_copy, name, (center[0] - 20, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    return frame_copy


# Example usage
if __name__ == "__main__":
    analyzer = SceneAnalyzer()
    
    # Simulate detections and tracks
    from object_detector import Detection
    from tracker import Track
    
    # Create fake frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create detections
    detections = [
        Detection(0, 'person', 0.9, 100, 100, 150, 200, 5000, (125, 150)),
        Detection(0, 'person', 0.85, 200, 150, 250, 250, 5000, (225, 200)),
    ]
    
    # Analyze
    crowd_regions = analyzer.analyze_crowd_density(frame, detections)
    print(f"Crowd regions: {len(crowd_regions)}")
    
    # Draw
    frame = draw_crowd_regions(frame, crowd_regions)
    
    cv2.imshow('Scene Analysis', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
