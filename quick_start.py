"""
Quick Start Script
Simplified setup and execution for rapid deployment
"""

import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_requirements():
    """Check if all requirements are installed"""
    logger.info("Checking requirements...")
    
    required_packages = [
        'cv2',
        'ultralytics',
        'torch',
        'torchvision',
        'numpy',
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package}")
        except ImportError:
            logger.warning(f"✗ {package}")
            missing.append(package)
    
    if missing:
        logger.error(f"\nMissing packages: {', '.join(missing)}")
        logger.info(f"Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    else:
        logger.info("\n✓ All requirements met!")
    
    return len(missing) == 0


def download_models():
    """Download YOLOv8 models if not present"""
    logger.info("\nChecking models...")
    
    model_path = Path("models")
    model_path.mkdir(exist_ok=True)
    
    try:
        from ultralytics import YOLO
        
        model_name = 'yolov8m.pt'
        logger.info(f"Loading {model_name}...")
        model = YOLO(model_name)
        logger.info(f"✓ {model_name} ready")
        
    except Exception as e:
        logger.error(f"Error downloading models: {e}")
        return False
    
    return True


def test_camera():
    """Test camera/video source"""
    logger.info("\nTesting video source...")
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Cannot open camera")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            logger.info(f"✓ Camera works! (Resolution: {frame.shape[1]}x{frame.shape[0]})")
            return True
        else:
            logger.error("Camera does not return frames")
            return False
    
    except Exception as e:
        logger.error(f"Camera test error: {e}")
        return False


def test_detection():
    """Quick detection test"""
    logger.info("\nRunning quick detection test...")
    
    try:
        import cv2
        import numpy as np
        from object_detector import ObjectDetector
        
        # Create dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Initialize detector
        detector = ObjectDetector()
        
        # Run detection
        detections = detector.detect(frame)
        
        logger.info(f"✓ Detection works! (Tested on {frame.shape})")
        return True
    
    except Exception as e:
        logger.error(f"Detection test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_run(video_source=0, max_frames=100):
    """Quick run with minimal configuration"""
    logger.info(f"\nStarting quick run (source={video_source}, max_frames={max_frames})...")
    
    try:
        from main import SafetyDetectionPipeline
        
        pipeline = SafetyDetectionPipeline(video_source=video_source)
        pipeline.run(max_frames=max_frames)
        
    except Exception as e:
        logger.error(f"Run error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def interactive_setup():
    """Interactive setup wizard"""
    print("\n" + "="*60)
    print("SAFETY DETECTION PIPELINE - QUICK START")
    print("="*60 + "\n")
    
    # Check requirements
    if not check_requirements():
        logger.error("Failed to install requirements")
        return False
    
    # Download models
    if not download_models():
        logger.error("Failed to download models")
        return False
    
    # Test camera
    if not test_camera():
        logger.warning("Camera test failed - trying with video file instead")
        while True:
            video_path = input("\nEnter video file path (or 'skip' to use webcam): ").strip()
            if video_path.lower() == 'skip':
                video_source = 0
                break
            elif Path(video_path).exists():
                video_source = video_path
                break
            else:
                logger.error("File not found")
    else:
        video_source = 0
    
    # Test detection
    if not test_detection():
        logger.error("Detection test failed")
        response = input("\nContinue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return False
    
    # Run
    print("\n" + "="*60)
    print("READY TO START")
    print("="*60)
    print(f"Video source: {video_source}")
    print("Controls:")
    print("  'Q' - Quit")
    print("  'P' - Pause/Resume")
    print("="*60 + "\n")
    
    response = input("Ready? Press 'y' to start or 'n' to cancel: ").strip().lower()
    if response == 'y':
        quick_run(video_source=video_source, max_frames=None)
    else:
        logger.info("Cancelled")
    
    return True


def headless_setup():
    """Non-interactive setup for automated deployment"""
    logger.info("Running headless setup...")
    
    if not check_requirements():
        return False
    
    if not download_models():
        return False
    
    if not test_detection():
        return False
    
    logger.info("✓ Headless setup complete")
    return True


def show_menu():
    """Show options menu"""
    print("\n" + "="*60)
    print("URBAN SAFETY DETECTION PIPELINE")
    print("="*60)
    print("\nOptions:")
    print("1. Interactive Setup & Run")
    print("2. Headless Setup Only")
    print("3. Run (Existing Config)")
    print("4. Run with Custom Video")
    print("5. Test Detection")
    print("6. Test Camera")
    print("7. Exit")
    print("="*60)


def main():
    """Main entry point"""
    
    while True:
        show_menu()
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == '1':
            interactive_setup()
        
        elif choice == '2':
            headless_setup()
        
        elif choice == '3':
            quick_run(video_source=0, max_frames=None)
        
        elif choice == '4':
            video_path = input("Enter video file path: ").strip()
            if Path(video_path).exists():
                quick_run(video_source=video_path, max_frames=None)
            else:
                logger.error("File not found")
        
        elif choice == '5':
            test_detection()
        
        elif choice == '6':
            test_camera()
        
        elif choice == '7':
            logger.info("Goodbye!")
            break
        
        else:
            logger.error("Invalid option")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode
        command = sys.argv[1]
        
        if command == '--setup':
            headless_setup()
        elif command == '--test':
            test_detection()
        elif command == '--run':
            video_source = sys.argv[2] if len(sys.argv) > 2 else 0
            try:
                video_source = int(video_source)
            except ValueError:
                pass
            quick_run(video_source=video_source, max_frames=None)
        else:
            print("Usage:")
            print("  python quick_start.py          - Interactive menu")
            print("  python quick_start.py --setup  - Setup only")
            print("  python quick_start.py --test   - Test detection")
            print("  python quick_start.py --run    - Run with webcam")
            print("  python quick_start.py --run <video> - Run with video file")
    else:
        # Interactive mode
        main()
