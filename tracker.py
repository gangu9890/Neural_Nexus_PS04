"""
Multi-Object Tracking Module using ByteTrack
Tracks movement of detected objects across frames
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from object_detector import Detection
from config import TRACKING_CONFIG, LOGGING_CONFIG

# Setup logging
logging.basicConfig(level=LOGGING_CONFIG['log_level'])
logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Represents a tracked object"""
    track_id: int
    class_name: str
    detections: List[Detection] = field(default_factory=list)
    positions: List[Tuple[int, int]] = field(default_factory=list)
    timestamps: List[int] = field(default_factory=list)
    
    # Tracking state
    consecutive_detections: int = 0
    frames_without_detection: int = 0
    is_active: bool = False
    
    # Movement statistics
    total_distance: float = 0.0
    max_velocity: float = 0.0
    direction: str = "stationary"  # left, right, up, down, diagonal
    
    def __post_init__(self):
        """Initialize track properties"""
        if self.detections:
            self.add_detection(self.detections[0], 0)
    
    def add_detection(self, detection: Detection, frame_idx: int):
        """Add detection to track"""
        self.detections.append(detection)
        self.positions.append(detection.center)
        self.timestamps.append(frame_idx)
        self.consecutive_detections += 1
        self.frames_without_detection = 0
        
        # Update movement statistics
        if len(self.positions) > 1:
            prev_pos = self.positions[-2]
            curr_pos = self.positions[-1]
            distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                             (curr_pos[1] - prev_pos[1])**2)
            self.total_distance += distance
            self.max_velocity = max(self.max_velocity, distance)
            self._update_direction(prev_pos, curr_pos)
    
    def _update_direction(self, prev_pos: Tuple[int, int], 
                         curr_pos: Tuple[int, int]):
        """Update movement direction"""
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]
        
        if abs(dx) < 2 and abs(dy) < 2:
            self.direction = "stationary"
        elif abs(dx) > abs(dy):
            self.direction = "right" if dx > 0 else "left"
        else:
            self.direction = "down" if dy > 0 else "up"
    
    def no_detection(self):
        """Update when no detection in frame"""
        self.frames_without_detection += 1
        self.consecutive_detections = 0
    
    def get_velocity(self) -> float:
        """Get average velocity"""
        if len(self.timestamps) < 2:
            return 0.0
        
        time_diff = self.timestamps[-1] - self.timestamps[0]
        if time_diff == 0:
            return 0.0
        
        distance = self.total_distance
        return distance / time_diff
    
    def get_current_position(self) -> Optional[Tuple[int, int]]:
        """Get latest position"""
        return self.positions[-1] if self.positions else None
    
    def get_bbox(self) -> Optional[Detection]:
        """Get latest bounding box"""
        return self.detections[-1] if self.detections else None
    
    def get_trajectory(self, last_n: int = None) -> List[Tuple[int, int]]:
        """Get trajectory (last n positions)"""
        if last_n is None:
            return self.positions.copy()
        return self.positions[-last_n:] if len(self.positions) > 0 else []


class SimpleTracker:
    """
    Simple tracker based on centroid matching
    More efficient than DeepSORT, good for real-time applications
    """
    
    def __init__(self, max_age: int = None, min_hits: int = None):
        """
        Initialize tracker
        
        Args:
            max_age: Max frames to keep track without detection
            min_hits: Min consecutive detections to confirm track
        """
        self.max_age = max_age or TRACKING_CONFIG['max_age']
        self.min_hits = min_hits or TRACKING_CONFIG['min_hits']
        self.iou_threshold = TRACKING_CONFIG['iou_threshold']
        
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.frame_count = 0
        
        logger.info(f"Initialized SimpleTracker: max_age={self.max_age}, "
                   f"min_hits={self.min_hits}")
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of detections from current frame
            
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Match detections to existing tracks
        matched_pairs, unmatched_dets, unmatched_trks = self._match_detections(
            detections
        )
        
        # Update matched tracks
        for track_id, det_idx in matched_pairs:
            track = self.tracks[track_id]
            track.add_detection(detections[det_idx], self.frame_count)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            detection = detections[det_idx]
            new_track = Track(
                track_id=self.next_track_id,
                class_name=detection.class_name,
                detections=[detection],
            )
            new_track.add_detection(detection, self.frame_count)
            self.tracks[self.next_track_id] = new_track
            self.next_track_id += 1
        
        # Update unmatched tracks (no detection in this frame)
        for track_id in unmatched_trks:
            track = self.tracks[track_id]
            track.no_detection()
        
        # Remove dead tracks
        self._remove_dead_tracks()
        
        # Return active tracks
        return self._get_active_tracks()
    
    def _match_detections(self, detections: List[Detection]) -> Tuple[
        List[Tuple[int, int]], List[int], List[int]
    ]:
        """
        Match detections to existing tracks using centroid distance
        
        Returns:
            Tuple of (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        matched_pairs = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(self.tracks.keys())
        
        if len(self.tracks) == 0 or len(detections) == 0:
            return matched_pairs, unmatched_dets, unmatched_trks
        
        # Calculate distances between detections and tracks
        distances = np.zeros((len(detections), len(self.tracks)))
        
        track_ids = list(self.tracks.keys())
        
        for det_idx, detection in enumerate(detections):
            for trk_idx, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                track_pos = track.get_current_position()
                
                if track_pos is None:
                    distances[det_idx, trk_idx] = float('inf')
                else:
                    # Euclidean distance between centroids
                    dist = np.sqrt(
                        (detection.center[0] - track_pos[0])**2 +
                        (detection.center[1] - track_pos[1])**2
                    )
                    distances[det_idx, trk_idx] = dist
        
        # Hungarian algorithm (simple greedy matching)
        matched_indices = []
        while True:
            # Find minimum distance
            min_idx = np.unravel_index(np.argmin(distances), distances.shape)
            min_dist = distances[min_idx]
            
            # Stop if minimum distance exceeds threshold
            max_distance = 50  # pixels
            if min_dist > max_distance:
                break
            
            det_idx, trk_idx = min_idx
            track_id = track_ids[trk_idx]
            
            matched_pairs.append((track_id, det_idx))
            unmatched_dets.remove(det_idx)
            unmatched_trks.remove(track_id)
            
            # Mark as used
            distances[det_idx, :] = float('inf')
            distances[:, trk_idx] = float('inf')
        
        return matched_pairs, unmatched_dets, unmatched_trks
    
    def _remove_dead_tracks(self):
        """Remove tracks that are too old"""
        dead_track_ids = []
        
        for track_id, track in self.tracks.items():
            # Remove if not active long enough and too many missed frames
            if (not track.is_active and 
                track.frames_without_detection > self.max_age):
                dead_track_ids.append(track_id)
            
            # Update active status
            if track.consecutive_detections >= self.min_hits:
                track.is_active = True
        
        for track_id in dead_track_ids:
            del self.tracks[track_id]
    
    def _get_active_tracks(self) -> List[Track]:
        """Get list of active tracks"""
        return [track for track in self.tracks.values() if track.is_active]
    
    def get_all_tracks(self) -> List[Track]:
        """Get all tracks (active and inactive)"""
        return list(self.tracks.values())
    
    def reset(self):
        """Reset tracker"""
        self.tracks.clear()
        self.frame_count = 0
        logger.info("Tracker reset")


class ByteTrackWrapper(SimpleTracker):
    """
    Wrapper around SimpleTracker with ByteTrack-like behavior
    Uses confidence-based association for better matching
    """
    
    def __init__(self, max_age: int = None, min_hits: int = None):
        """Initialize ByteTrack wrapper"""
        super().__init__(max_age, min_hits)
        logger.info("Using ByteTrack-like tracking")
    
    def _match_detections(self, detections: List[Detection]) -> Tuple[
        List[Tuple[int, int]], List[int], List[int]
    ]:
        """
        Match detections using confidence-weighted distance
        Higher confidence detections get priority
        """
        # Sort by confidence
        sorted_dets = sorted(enumerate(detections), 
                           key=lambda x: x[1].confidence, reverse=True)
        
        matched_pairs = []
        unmatched_dets = [i for i, _ in sorted_dets]
        unmatched_trks = list(self.tracks.keys())
        
        if len(self.tracks) == 0 or len(detections) == 0:
            return matched_pairs, unmatched_dets, unmatched_trks
        
        track_ids = list(self.tracks.keys())
        
        # Match high-confidence detections first
        for det_orig_idx, detection in sorted_dets:
            best_track_id = None
            best_distance = 50  # max distance threshold
            
            for track_id in unmatched_trks:
                track = self.tracks[track_id]
                track_pos = track.get_current_position()
                
                if track_pos is None:
                    continue
                
                # Distance with confidence boost
                dist = np.sqrt(
                    (detection.center[0] - track_pos[0])**2 +
                    (detection.center[1] - track_pos[1])**2
                )
                
                # Confidence weighted distance
                weighted_dist = dist / (1.0 + detection.confidence)
                
                if weighted_dist < best_distance:
                    best_distance = weighted_dist
                    best_track_id = track_id
            
            if best_track_id is not None:
                matched_pairs.append((best_track_id, det_orig_idx))
                unmatched_dets.remove(det_orig_idx)
                unmatched_trks.remove(best_track_id)
        
        return matched_pairs, unmatched_dets, unmatched_trks


def create_tracker(tracker_type: str = None, 
                  max_age: int = None,
                  min_hits: int = None) -> SimpleTracker:
    """
    Factory function to create tracker
    
    Args:
        tracker_type: 'bytetrack' or 'simple'
        max_age: Max frames without detection
        min_hits: Min hits to confirm
        
    Returns:
        Tracker instance
    """
    tracker_type = tracker_type or TRACKING_CONFIG['tracker_type']
    
    if tracker_type == 'bytetrack':
        return ByteTrackWrapper(max_age, min_hits)
    else:
        return SimpleTracker(max_age, min_hits)


def draw_tracks(frame: np.ndarray, tracks: List[Track],
               draw_trajectory: bool = True,
               draw_id: bool = True) -> np.ndarray:
    """
    Draw tracks on frame
    
    Args:
        frame: Input frame
        tracks: List of tracks
        draw_trajectory: Whether to draw trajectory
        draw_id: Whether to draw track ID
        
    Returns:
        Frame with drawn tracks
    """
    frame_copy = frame.copy()
    
    colors = {}  # Cache colors for each track ID
    
    for track in tracks:
        track_id = track.track_id
        
        # Assign color
        if track_id not in colors:
            colors[track_id] = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
            )
        color = colors[track_id]
        
        # Draw current bounding box
        bbox = track.get_bbox()
        if bbox:
            cv2.rectangle(frame_copy, (bbox.x1, bbox.y1), 
                         (bbox.x2, bbox.y2), color, 2)
            
            if draw_id:
                text = f"ID:{track_id}"
                cv2.putText(frame_copy, text, (bbox.x1, bbox.y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw trajectory
        if draw_trajectory:
            trajectory = track.get_trajectory(last_n=20)
            for i in range(1, len(trajectory)):
                pt1 = trajectory[i - 1]
                pt2 = trajectory[i]
                cv2.line(frame_copy, pt1, pt2, color, 1)
    
    return frame_copy


# Example usage
if __name__ == "__main__":
    # Initialize tracker
    tracker = create_tracker('bytetrack')
    
    # Simulate detections
    from object_detector import Detection
    
    det1 = Detection(
        class_id=0, class_name='person', confidence=0.9,
        x1=100, y1=100, x2=150, y2=200, bbox_area=5000, center=(125, 150)
    )
    
    detections = [det1]
    
    # Update tracker
    tracks = tracker.update(detections)
    
    print(f"Active tracks: {len(tracks)}")
    for track in tracks:
        print(f"  Track {track.track_id}: {track.class_name}, "
              f"positions={len(track.positions)}, "
              f"velocity={track.get_velocity():.2f}")
