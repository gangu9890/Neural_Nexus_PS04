"""
Frame Extractor Module
Handles video input and frame extraction from various sources
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Iterator
from pathlib import Path
import logging
from config import VIDEO_CONFIG, LOGGING_CONFIG

# Setup logging
logging.basicConfig(level=LOGGING_CONFIG['log_level'])
logger = logging.getLogger(__name__)


class FrameExtractor:
    """
    Extracts frames from video sources (webcam, video file, image sequence)
    """
    
    def __init__(self, source: any = None):
        """
        Initialize frame extractor
        
        Args:
            source: Video file path, webcam index (0, 1, ...), or None for default
        """
        self.source = source if source is not None else VIDEO_CONFIG['input_source']
        self.cap = None
        self.frame_count = 0
        self.skip_frames = VIDEO_CONFIG['skip_frames']
        self.target_fps = VIDEO_CONFIG['fps']
        self.target_width = VIDEO_CONFIG['frame_width']
        self.target_height = VIDEO_CONFIG['frame_height']
        
        self._initialize_capture()
    
    def _initialize_capture(self):
        """Initialize video capture from source"""
        try:
            if isinstance(self.source, str):
                # Video file
                if not Path(self.source).exists():
                    raise FileNotFoundError(f"Video file not found: {self.source}")
                logger.info(f"Opening video file: {self.source}")
            else:
                # Webcam
                logger.info(f"Opening webcam: {self.source}")
            
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video source: {self.source}")
            
            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video Properties:")
            logger.info(f"  Resolution: {self.width}x{self.height}")
            logger.info(f"  FPS: {self.fps}")
            logger.info(f"  Total Frames: {self.total_frames}")
            
        except Exception as e:
            logger.error(f"Error initializing capture: {e}")
            raise
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get next frame with preprocessing
        
        Returns:
            Tuple of (success, frame)
                - success: bool indicating if frame was read
                - frame: numpy array of shape (height, width, 3) or None
        """
        if self.cap is None:
            logger.error("Capture not initialized")
            return False, None
        
        # Skip frames if needed
        for _ in range(self.skip_frames):
            ret = self.cap.grab()  # Skip without decoding
            if not ret:
                return False, None
        
        ret, frame = self.cap.read()
        
        if not ret:
            logger.warning("Failed to read frame or reached end of video")
            return False, None
        
        # Preprocess frame
        frame = self._preprocess_frame(frame)
        self.frame_count += 1
        
        return True, frame
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame: resize, normalize
        
        Args:
            frame: Raw frame from video
            
        Returns:
            Preprocessed frame
        """
        # Resize to target resolution
        frame = cv2.resize(frame, (self.target_width, self.target_height), 
                          interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB (if needed for some models)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame
    
    def get_frames_batch(self, batch_size: int = 1) -> Iterator[Tuple[bool, list]]:
        """
        Generator yielding batches of frames
        
        Args:
            batch_size: Number of frames per batch
            
        Yields:
            Tuple of (all_success, frames_list)
        """
        batch = []
        while True:
            success, frame = self.get_frame()
            
            if success:
                batch.append(frame)
                if len(batch) == batch_size:
                    yield True, batch
                    batch = []
            else:
                if batch:  # Yield remaining frames
                    yield True, batch
                yield False, []
                break
    
    def seek(self, frame_index: int) -> bool:
        """
        Seek to specific frame
        
        Args:
            frame_index: Frame number to seek to
            
        Returns:
            Success status
        """
        if self.cap is None:
            return False
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self.frame_count = frame_index
        return True
    
    def get_position(self) -> float:
        """Get current video position in milliseconds"""
        if self.cap is None:
            return 0.0
        return self.cap.get(cv2.CAP_PROP_POS_MSEC)
    
    def get_current_frame_index(self) -> int:
        """Get current frame index"""
        return self.frame_count
    
    def reset(self):
        """Reset to beginning of video"""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_count = 0
    
    def release(self):
        """Release video capture resource"""
        if self.cap:
            self.cap.release()
            logger.info("Video capture released")
    
    def __iter__(self):
        """Iterator support"""
        self.reset()
        return self
    
    def __next__(self) -> Tuple[bool, np.ndarray]:
        """Get next frame when used as iterator"""
        success, frame = self.get_frame()
        if not success:
            raise StopIteration
        return success, frame
    
    def __del__(self):
        """Cleanup on deletion"""
        self.release()
    
    def get_video_info(self) -> dict:
        """Get video information"""
        return {
            'source': self.source,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'duration_seconds': self.total_frames / self.fps if self.fps > 0 else 0,
            'current_frame': self.frame_count,
        }


class VideoWriter:
    """
    Writes frames to video file
    """
    
    def __init__(self, output_path: str, frame_width: int, frame_height: int, 
                 fps: int = 30, codec: str = 'mp4v'):
        """
        Initialize video writer
        
        Args:
            output_path: Path to save video
            frame_width: Frame width
            frame_height: Frame height
            fps: Frames per second
            codec: Video codec
        """
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, 
                                     (frame_width, frame_height))
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot create video writer: {output_path}")
        
        logger.info(f"Video writer initialized: {output_path}")
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write frame to video
        
        Args:
            frame: Frame to write
            
        Returns:
            Success status
        """
        return self.writer.write(frame)
    
    def release(self):
        """Release video writer"""
        if self.writer:
            self.writer.release()
            logger.info(f"Video saved: {self.output_path}")
    
    def __del__(self):
        self.release()


# Example usage
if __name__ == "__main__":
    # Test with webcam
    extractor = FrameExtractor(source=0)  # 0 for webcam
    
    print("Video Info:", extractor.get_video_info())
    
    # Get first 10 frames
    for i in range(10):
        success, frame = extractor.get_frame()
        if success:
            print(f"Frame {i}: shape={frame.shape}, dtype={frame.dtype}")
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    extractor.release()
    cv2.destroyAllWindows()
