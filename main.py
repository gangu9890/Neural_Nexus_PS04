"""
Main Pipeline Orchestrator
Coordinates all modules: frame extraction, detection, tracking, analysis, and classification
"""

import cv2
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime

from frame_extractor import FrameExtractor, VideoWriter
from object_detector import ObjectDetector
from tracker import create_tracker, draw_tracks
from scene_analyzer import SceneAnalyzer, draw_crowd_regions, draw_anomalies, draw_restricted_zones
from event_classifier import EventClassifier, draw_incidents
from config import (
    VIDEO_CONFIG, YOLO_CONFIG, VISUALIZATION_CONFIG, LOGGING_CONFIG,
    GEOLOCATION_CONFIG, DRONE_DISPATCH_CONFIG, RESTRICTED_ZONES
)

# Setup logging
logging.basicConfig(
    level=LOGGING_CONFIG['log_level'],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SafetyDetectionPipeline:
    """
    Complete pipeline for urban safety detection and incident classification
    """
    
    def __init__(self, video_source: any = None, output_video: str = None):
        """
        Initialize pipeline
        
        Args:
            video_source: Video file path or webcam index
            output_video: Path to save output video
        """
        logger.info("Initializing Safety Detection Pipeline...")
        
        # Initialize modules
        self.frame_extractor = FrameExtractor(video_source)
        self.detector = ObjectDetector()
        self.tracker = create_tracker()
        self.scene_analyzer = SceneAnalyzer()
        self.event_classifier = EventClassifier()
        
        # Output
        self.output_video = output_video
        self.video_writer = None
        if output_video:
            video_info = self.frame_extractor.get_video_info()
            self.video_writer = VideoWriter(
                output_video,
                video_info['width'],
                video_info['height'],
                fps=video_info['fps']
            )
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'total_tracks': 0,
            'total_incidents': 0,
            'incidents_by_type': {},
        }
        
        logger.info("Pipeline initialized successfully")
    
    def run(self, max_frames: int = None, show_fps: bool = True):
        """
        Run pipeline on video
        
        Args:
            max_frames: Maximum frames to process (None for all)
            show_fps: Whether to display FPS
        """
        logger.info("Starting pipeline execution...")
        
        frame_count = 0
        detection_times = []
        tracking_times = []
        analysis_times = []
        
        while True:
            # Check frame limit
            if max_frames and frame_count >= max_frames:
                logger.info(f"Reached max frames limit: {max_frames}")
                break
            
            # Get frame
            success, frame = self.frame_extractor.get_frame()
            if not success:
                logger.info("End of video reached")
                break
            
            frame_count += 1
            
            # === DETECTION ===
            import time
            t0 = time.time()
            detections = self.detector.detect(frame)
            detection_times.append(time.time() - t0)
            
            # === TRACKING ===
            t0 = time.time()
            tracks = self.tracker.update(detections)
            tracking_times.append(time.time() - t0)
            
            # === SCENE ANALYSIS ===
            t0 = time.time()
            
            crowd_regions = self.scene_analyzer.analyze_crowd_density(
                frame, detections
            )
            anomalies = self.scene_analyzer.detect_anomalies(tracks)
            unauthorized_entries = self.scene_analyzer.detect_unauthorized_entry(
                frame, detections, RESTRICTED_ZONES
            )
            vehicle_stops = self.scene_analyzer.detect_vehicle_stops(tracks)
            
            analysis_times.append(time.time() - t0)
            
            # === EVENT CLASSIFICATION ===
            frame_h, frame_w = frame.shape[:2]
            new_incidents = self.event_classifier.classify_incident(
                frame_h, frame_w,
                detections, tracks,
                crowd_regions, anomalies,
                unauthorized_entries, vehicle_stops
            )
            
            # === VISUALIZATION ===
            display_frame = self._draw_results(
                frame, detections, tracks, crowd_regions,
                anomalies, unauthorized_entries, new_incidents
            )
            
            # Add FPS
            if show_fps and len(detection_times) > 0:
                avg_fps = 1.0 / (np.mean(detection_times[-10:]) + 
                                np.mean(tracking_times[-10:]) +
                                np.mean(analysis_times[-10:]))
                cv2.putText(display_frame, f"FPS: {avg_fps:.1f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
            
            # Add frame info
            cv2.putText(display_frame, f"Frame: {frame_count}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 0), 1)
            
            # === OUTPUT ===
            cv2.imshow('Safety Detection Pipeline', display_frame)
            
            if self.video_writer:
                self.video_writer.write_frame(display_frame)
            
            # Print incidents
            if new_incidents:
                logger.warning(f"Frame {frame_count}: {len(new_incidents)} incident(s) detected")
                for incident in new_incidents:
                    logger.warning(f"  {incident}")
            
            # Handle user input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("User quit")
                break
            elif key == ord('p'):
                logger.info("Paused - press any key to continue")
                cv2.waitKey(0)
            
            # Update statistics
            self.stats['frames_processed'] += 1
            self.stats['total_detections'] += len(detections)
            self.stats['total_tracks'] += len(tracks)
            self.stats['total_incidents'] += len(new_incidents)
            
            for incident in new_incidents:
                incident_type = incident.incident_type
                self.stats['incidents_by_type'][incident_type] = \
                    self.stats['incidents_by_type'].get(incident_type, 0) + 1
            
            # Print progress
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames | "
                           f"Detections: {len(detections)} | "
                           f"Tracks: {len(tracks)} | "
                           f"Incidents: {self.event_classifier.get_incident_summary()}")
        
        # Cleanup
        self._cleanup()
        
        # Print final statistics
        self._print_statistics()
    
    def _draw_results(self, frame, detections, tracks, crowd_regions,
                     anomalies, unauthorized_entries, incidents):
        """Draw all results on frame"""
        display_frame = frame.copy()
        
        # Draw detections
        if VISUALIZATION_CONFIG['show_detections']:
            display_frame = self.detector.draw_detections(
                display_frame, detections
            )
        
        # Draw tracks
        if VISUALIZATION_CONFIG['show_tracks']:
            display_frame = draw_tracks(display_frame, tracks)
        
        # Draw crowd regions
        if VISUALIZATION_CONFIG['show_crowd_heatmap']:
            display_frame = draw_crowd_regions(display_frame, crowd_regions)
        
        # Draw anomalies
        if VISUALIZATION_CONFIG['show_anomalies']:
            display_frame = draw_anomalies(display_frame, anomalies, tracks)
        
        # Draw restricted zones
        display_frame = draw_restricted_zones(display_frame, RESTRICTED_ZONES)
        
        # Draw incidents
        display_frame = draw_incidents(display_frame, incidents)
        
        return display_frame
    
    def _cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up...")
        self.frame_extractor.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
    
    def _print_statistics(self):
        """Print execution statistics"""
        logger.info("\n" + "="*60)
        logger.info("EXECUTION STATISTICS")
        logger.info("="*60)
        logger.info(f"Total frames processed: {self.stats['frames_processed']}")
        logger.info(f"Total detections: {self.stats['total_detections']}")
        logger.info(f"Total tracks: {self.stats['total_tracks']}")
        logger.info(f"Total incidents: {self.stats['total_incidents']}")
        
        if self.stats['frames_processed'] > 0:
            avg_detections = self.stats['total_detections'] / self.stats['frames_processed']
            avg_tracks = self.stats['total_tracks'] / self.stats['frames_processed']
            logger.info(f"\nAverage per frame:")
            logger.info(f"  Detections: {avg_detections:.2f}")
            logger.info(f"  Tracks: {avg_tracks:.2f}")
        
        logger.info(f"\nIncidents by type:")
        for incident_type, count in self.stats['incidents_by_type'].items():
            logger.info(f"  {incident_type}: {count}")
        
        logger.info("="*60 + "\n")
    
    def get_statistics(self):
        """Get pipeline statistics"""
        return self.stats.copy()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Urban Safety Detection Pipeline'
    )
    parser.add_argument('--video', type=str, default=None,
                       help='Video file path or camera index (default: 0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video file path')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process')
    parser.add_argument('--no-fps', action='store_true',
                       help='Do not display FPS')
    
    args = parser.parse_args()
    
    # Determine video source
    video_source = 0  # Default to webcam
    if args.video is not None:
        try:
            video_source = int(args.video)  # Try as camera index
        except ValueError:
            video_source = args.video  # Use as file path
    
    logger.info(f"Video source: {video_source}")
    logger.info(f"Output video: {args.output}")
    
    # Create and run pipeline
    pipeline = SafetyDetectionPipeline(
        video_source=video_source,
        output_video=args.output
    )
    
    pipeline.run(
        max_frames=args.max_frames,
        show_fps=not args.no_fps
    )
    
    # Save statistics
    stats = pipeline.get_statistics()
    stats_file = 'pipeline_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Statistics saved to {stats_file}")


if __name__ == "__main__":
    main()
