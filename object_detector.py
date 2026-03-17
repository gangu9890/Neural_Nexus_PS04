"""
Object Detection Module using YOLOv8
Detects people, vehicles, and other objects in frames
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

from ultralytics import YOLO
from config import YOLO_CONFIG, DETECTION_CLASSES, LOGGING_CONFIG, PERFORMANCE_CONFIG

# Setup logging
logging.basicConfig(level=LOGGING_CONFIG['log_level'])
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Represents a single detection"""
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    bbox_area: int
    center: Tuple[int, int]
    
    def __post_init__(self):
        """Calculate additional properties"""
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        self.bbox_area = self.width * self.height


class ObjectDetector:
    """
    YOLOv8 based object detector
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize YOLOv8 detector
        
        Args:
            model_name: Model name (e.g., 'yolov8m.pt')
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name or YOLO_CONFIG['model_name']
        self.device = device or YOLO_CONFIG['device']
        
        logger.info(f"Loading YOLOv8 model: {self.model_name} on {self.device}")
        
        try:
            # Fix for PyTorch 2.6+ weights_only=True default
            try:
                import torch
                from ultralytics.nn.tasks import DetectionModel
                if hasattr(torch.serialization, 'add_safe_globals'):
                    torch.serialization.add_safe_globals([DetectionModel])
                    logger.info("Added DetectionModel to PyTorch safe globals")
            except Exception as e:
                logger.warning(f"Failed to add safe globals, model load might fail on PyTorch 2.6+: {e}")

            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            
            # Get COCO class names
            self.class_names = self.model.names
            
            logger.info(f"Model loaded successfully. Classes: {len(self.class_names)}")
            logger.info(f"Available classes: {list(self.class_names.values())[:10]}...")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        self.inference_time = 0
    
    def detect(self, frame: np.ndarray, 
               conf_threshold: float = None) -> List[Detection]:
        """
        Run object detection on frame
        
        Args:
            frame: Input frame (BGR numpy array)
            conf_threshold: Confidence threshold (uses config if None)
            
        Returns:
            List of Detection objects
        """
        conf_threshold = conf_threshold or YOLO_CONFIG['confidence_threshold']
        
        try:
            # Run inference
            with (torch.inference_mode() if PERFORMANCE_CONFIG['inference_mode'] 
                  else torch.enable_grad()):
                results = self.model.predict(
                    source=frame,
                    conf=conf_threshold,
                    iou=YOLO_CONFIG['iou_threshold'],
                    device=self.device,
                    verbose=False,
                    max_det=YOLO_CONFIG['max_det'],
                )
            
            detections = self._parse_results(results[0], frame.shape)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []
    
    def _parse_results(self, result, frame_shape: Tuple) -> List[Detection]:
        """
        Parse YOLOv8 results into Detection objects
        
        Args:
            result: YOLO prediction result
            frame_shape: Shape of input frame (height, width, channels)
            
        Returns:
            List of Detection objects
        """
        detections = []
        frame_h, frame_w = frame_shape[:2]
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        for box in result.boxes:
            # Extract box coordinates
            xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = xyxy.astype(int)
            
            # Clamp to frame bounds
            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(0, min(x2, frame_w - 1))
            y2 = max(0, min(y2, frame_h - 1))
            
            # Extract class and confidence
            class_id = int(box.cls[0].cpu().item())
            confidence = float(box.conf[0].cpu().item())
            class_name = self.class_names.get(class_id, f"Class_{class_id}")
            
            # Calculate center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            detection = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                bbox_area=(x2 - x1) * (y2 - y1),
                center=(center_x, center_y),
            )
            
            detections.append(detection)
        
        return detections
    
    def filter_by_class(self, detections: List[Detection], 
                       class_names: List[str]) -> List[Detection]:
        """
        Filter detections by class names
        
        Args:
            detections: List of detections
            class_names: Target class names
            
        Returns:
            Filtered detections
        """
        return [d for d in detections if d.class_name in class_names]
    
    def filter_by_confidence(self, detections: List[Detection], 
                            min_conf: float) -> List[Detection]:
        """
        Filter detections by minimum confidence
        
        Args:
            detections: List of detections
            min_conf: Minimum confidence threshold
            
        Returns:
            Filtered detections
        """
        return [d for d in detections if d.confidence >= min_conf]
    
    def filter_by_area(self, detections: List[Detection], 
                      min_area: int, max_area: int = None) -> List[Detection]:
        """
        Filter detections by bounding box area
        
        Args:
            detections: List of detections
            min_area: Minimum area in pixels
            max_area: Maximum area in pixels (None for no limit)
            
        Returns:
            Filtered detections
        """
        filtered = [d for d in detections if d.bbox_area >= min_area]
        if max_area is not None:
            filtered = [d for d in filtered if d.bbox_area <= max_area]
        return filtered
    
    def nms(self, detections: List[Detection], 
            iou_threshold: float = 0.3) -> List[Detection]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections
        
        Args:
            detections: List of detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Filtered detections after NMS
        """
        if len(detections) < 2:
            return detections
        
        # Convert to format for cv2.dnn.NMSBoxes
        boxes = [(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1) for d in detections]
        confidences = [d.confidence for d in detections]
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes,
            scores=confidences,
            score_threshold=0.0,
            nms_threshold=iou_threshold,
        )
        
        if len(indices) == 0:
            return []
        
        return [detections[i] for i in indices.flatten()]
    
    def get_person_detections(self, detections: List[Detection]) -> List[Detection]:
        """Get only person detections"""
        return [d for d in detections if d.class_name == 'person']
    
    def get_vehicle_detections(self, detections: List[Detection]) -> List[Detection]:
        """Get vehicle detections (car, truck, bus, etc.)"""
        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
        return [d for d in detections if d.class_name in vehicle_classes]
    
    def get_object_detections(self, detections: List[Detection]) -> List[Detection]:
        """Get object detections (backpack, luggage, etc.)"""
        object_classes = ['backpack', 'handbag', 'suitcase', 'umbrella', 'tie']
        return [d for d in detections if d.class_name in object_classes]
    
    def draw_detections(self, frame: np.ndarray, 
                       detections: List[Detection],
                       draw_center: bool = False,
                       text_thickness: int = 1) -> np.ndarray:
        """
        Draw detections on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            draw_center: Whether to draw center point
            text_thickness: Thickness of text
            
        Returns:
            Frame with drawn detections
        """
        frame_copy = frame.copy()
        
        for detection in detections:
            # Draw bounding box
            color = self._get_color_by_class(detection.class_name)
            cv2.rectangle(frame_copy, (detection.x1, detection.y1), 
                         (detection.x2, detection.y2), color, 2)
            
            # Draw label
            label = f"{detection.class_name} {detection.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.6, text_thickness)
            
            cv2.rectangle(frame_copy, 
                         (detection.x1, detection.y1 - label_size[1] - 4),
                         (detection.x1 + label_size[0], detection.y1),
                         color, -1)
            
            cv2.putText(frame_copy, label, 
                       (detection.x1, detection.y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 
                       text_thickness)
            
            # Draw center point
            if draw_center:
                cv2.circle(frame_copy, detection.center, 3, (0, 255, 0), -1)
        
        return frame_copy
    
    @staticmethod
    def _get_color_by_class(class_name: str) -> Tuple[int, int, int]:
        """Get color for class name"""
        color_map = {
            'person': (0, 255, 0),      # Green
            'car': (0, 0, 255),         # Red
            'truck': (0, 165, 255),     # Orange
            'bus': (255, 0, 0),         # Blue
            'motorcycle': (255, 0, 255),  # Magenta
            'bicycle': (0, 255, 255),   # Yellow
            'backpack': (128, 0, 128),  # Purple
            'handbag': (128, 128, 0),   # Olive
        }
        return color_map.get(class_name, (128, 128, 128))  # Gray as default
    
    def get_stats(self, detections: List[Detection]) -> Dict:
        """Get statistics about detections"""
        if not detections:
            return {
                'total_detections': 0,
                'people_count': 0,
                'vehicles_count': 0,
                'objects_count': 0,
            }
        
        people = self.get_person_detections(detections)
        vehicles = self.get_vehicle_detections(detections)
        objects = self.get_object_detections(detections)
        
        return {
            'total_detections': len(detections),
            'people_count': len(people),
            'vehicles_count': len(vehicles),
            'objects_count': len(objects),
            'avg_confidence': np.mean([d.confidence for d in detections]),
        }


# Import torch for inference mode
try:
    import torch
except ImportError:
    logger.warning("PyTorch not found. Some features may not work.")


# Example usage
if __name__ == "__main__":
    import cv2
    
    # Initialize detector
    detector = ObjectDetector()
    
    # Test on webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize for faster processing
        frame = cv2.resize(frame, (640, 480))
        
        # Run detection
        detections = detector.detect(frame)
        
        # Print stats
        stats = detector.get_stats(detections)
        print(f"Frame detections: {stats['total_detections']}, People: {stats['people_count']}")
        
        # Draw detections
        frame = detector.draw_detections(frame, detections)
        
        # Display
        cv2.imshow('Detections', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
