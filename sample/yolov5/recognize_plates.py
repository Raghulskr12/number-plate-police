import cv2
import torch
import easyocr
import numpy as np
import json
from datetime import datetime
import time
import warnings
import os
import re
from collections import Counter
import logging
from PIL import Image, ImageEnhance, ImageFilter
import torch.nn.functional as F

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("anpr_log.txt"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
camera_id = "Gate_Camera_1"
duplicate_window = 5  # Seconds to consider duplicates
confidence_threshold = 0.5  # Increased from 0.4
model_dir = "."  # Current directory for model
output_dir = "../"  # Output directory
json_file = os.path.join(output_dir, "plates.json")
result_images_dir = os.path.join(output_dir, "result_images")
debug_dir = os.path.join(output_dir, "debug_images")

# Indian license plate patterns
indian_plate_patterns = [
    r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$',  # Standard format: MH12AB1234
    r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,1}[0-9]{4}$',   # Older format: MH12A1234
    r'^[A-Z]{3}[0-9]{4}$',                      # Diplomat plates: UN1234
    r'^[A-Z]{2}[0-9]{3,4}$',                    # Older format: MH1234
    r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{3,4}$'     # Newer format: DL01AB1234
]

# Create directories
os.makedirs(result_images_dir, exist_ok=True)
os.makedirs(debug_dir, exist_ok=True)

# Initialize the EasyOCR reader (more accurate than Tesseract for Indian plates)
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# Load YOLOv5 model
def load_model():
    try:
        # Try to load YOLOv5 model
        model = torch.hub.load(model_dir, 'custom', path='runs/train/exp/weights/best.pt', source='local')
        model.conf = confidence_threshold
        model.iou = 0.45
        
        # Enable GPU if available
        if torch.cuda.is_available():
            model.cuda()
            logger.info("Using GPU acceleration")
        else:
            logger.info("GPU not available, using CPU")
            
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        exit(1)

model = load_model()

class LicensePlateProcessor:
    """Class for processing license plates with multiple approaches"""
    
    @staticmethod
    def preprocess_for_ocr(plate_img):
        """Apply multiple preprocessing techniques"""
        
        # Convert to PIL Image for easier processing
        pil_img = Image.fromarray(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
        
        processed_images = []
        
        # Original image
        processed_images.append(np.array(pil_img.convert('L')))
        
        # 1. Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced_img = enhancer.enhance(2.0)
        processed_images.append(np.array(enhanced_img.convert('L')))
        
        # 2. Sharpen
        sharpened = pil_img.filter(ImageFilter.SHARPEN)
        processed_images.append(np.array(sharpened.convert('L')))
        
        # 3. OpenCV adaptiveThreshold
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Binary threshold
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        processed_images.append(binary)
        
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
        processed_images.append(adaptive)
        
        # 4. Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(adaptive, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        processed_images.append(eroded)
        
        # 5. Canny edge detection followed by dilation
        edges = cv2.Canny(gray, 100, 200)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        processed_images.append(dilated_edges)
        
        return processed_images
    
    @staticmethod
    def correct_plate_text(text):
        """Correct common OCR mistakes in Indian license plates"""
        corrections = {
            '0': 'O', 'Q': 'O', 'D': '0', # Common confusions
            'I': '1', 'L': '1',
            'Z': '2',
            'A': '4',
            'S': '5',
            'G': '6',
            'T': '7',
            'B': '8',
            'g': '9',
            'o': '0',
            'i': '1',
            'l': '1',
            's': '5',
            'z': '2'
        }
        
        # Apply corrections based on position in the plate
        if len(text) >= 4:
            # First two characters are typically letters
            for i in range(min(2, len(text))):
                if text[i].isdigit():
                    if text[i] == '0':
                        text = text[:i] + 'O' + text[i+1:]
                    elif text[i] == '1':
                        text = text[:i] + 'I' + text[i+1:]
                    elif text[i] == '8':
                        text = text[:i] + 'B' + text[i+1:]
            
            # Last four characters are typically digits
            for i in range(max(0, len(text) - 4), len(text)):
                if text[i].isalpha():
                    if text[i] in corrections:
                        text = text[:i] + corrections[text[i]] + text[i+1:]
        
        # Remove any spaces or special characters
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        return text
    
    @staticmethod
    def read_plate(plate_img, debug=False):
        """Read plate using EasyOCR with multiple preprocessing approaches"""
        processed_images = LicensePlateProcessor.preprocess_for_ocr(plate_img)
        
        all_results = []
        
        if debug:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_image_path = os.path.join(debug_dir, f"debug_{timestamp}.jpg")
            
            h, w = processed_images[0].shape
            grid = np.zeros((h * 2, w * 3), dtype=np.uint8)
            
            for i, img in enumerate(processed_images[:6]):
                r, c = i // 3, i % 3
                grid[r*h:(r+1)*h, c*w:(c+1)*w] = img
                
            cv2.imwrite(debug_image_path, grid)
        
        for idx, img in enumerate(processed_images):
            if len(img.shape) == 2:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = img
                
            results = reader.readtext(img_rgb, detail=1, paragraph=False)
            
            for (bbox, text, prob) in results:
                if prob > 0.3:
                    text = LicensePlateProcessor.correct_plate_text(text)
                    if 4 <= len(text) <= 10:
                        all_results.append((text, prob))
        
        if not all_results:
            results = reader.readtext(plate_img, detail=1, paragraph=False, 
                                     allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            for (bbox, text, prob) in results:
                if prob > 0.2:
                    text = LicensePlateProcessor.correct_plate_text(text)
                    if 4 <= len(text) <= 10:
                        all_results.append((text, prob))
        
        if not all_results:
            return "", 0.0
        
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        weighted_counts = {}
        for text, prob in all_results:
            if text in weighted_counts:
                weighted_counts[text] += prob
            else:
                weighted_counts[text] = prob
        
        if weighted_counts:
            most_likely = max(weighted_counts.items(), key=lambda x: x[1])
            return most_likely[0], float(most_likely[1])
        
        return "", 0.0
    
    @staticmethod
    def validate_plate(plate_text):
        """Validate if the plate text is likely to be a valid Indian license plate"""
        if not plate_text:
            return False
        
        if len(plate_text) < 4 or len(plate_text) > 10:
            return False
        
        for pattern in indian_plate_patterns:
            if re.match(pattern, plate_text):
                return True
        
        if len(plate_text) >= 4:
            if not (plate_text[0].isalpha() and plate_text[1].isalpha()):
                return False
            
            if not any(char.isdigit() for char in plate_text):
                return False
            
            if sum(1 for c in plate_text[2:] if c.isdigit()) < len(plate_text[2:]) / 2:
                return False
            
            return True
        
        return False

# Tracking class for duplicates
class PlateTracker:
    def __init__(self, time_window=5):
        self.recent_plates = []
        self.time_window = time_window
    
    def is_duplicate(self, plate_text):
        """Check if a plate was recently seen"""
        current_time = time.time()
        
        self.recent_plates = [entry for entry in self.recent_plates 
                              if (current_time - entry["time"]) < self.time_window]
        
        for entry in self.recent_plates:
            if entry["plate"] == plate_text:
                return True
        
        return False
    
    def add_plate(self, plate_text):
        """Add a plate to the tracker"""
        self.recent_plates.append({"plate": plate_text, "time": time.time()})

# Initialize plate tracker
plate_tracker = PlateTracker(duplicate_window)

def extract_plate_region(frame, bbox, padding=20):
    """Extract the plate region with padding"""
    x_min, y_min, x_max, y_max = [int(coord) for coord in bbox[:4]]
    
    height, width = frame.shape[:2]
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)
    
    return frame[y_min:y_max, x_min:x_max]

def process_frame(frame, frame_id=None, debug=False):
    """Process a single frame and return the license plate data"""
    display_frame = frame.copy()
    
    max_dim = 1280
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        display_frame = frame.copy()
    
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()
    
    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    
    frame_results = []
    
    for det in detections:
        x_min, y_min, x_max, y_max, conf, cls = det
        if conf > confidence_threshold:
            plate_img = extract_plate_region(frame, det)
            
            if plate_img.size == 0:
                continue
            
            plate_text, ocr_confidence = LicensePlateProcessor.read_plate(plate_img, debug)
            
            if plate_text and LicensePlateProcessor.validate_plate(plate_text):
                combined_confidence = (conf + ocr_confidence) / 2
                
                cv2.rectangle(display_frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), 
                             (0, 255, 0), 2)
                
                cv2.putText(display_frame, f"{plate_text} ({combined_confidence:.2f})", 
                           (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           (0, 255, 0), 2)
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                timestamp_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if not plate_tracker.is_duplicate(plate_text):
                    result_img_filename = f"{plate_text}_{timestamp_filename}.jpg"
                    result_img_path = os.path.join(result_images_dir, result_img_filename)
                    cv2.imwrite(result_img_path, plate_img)
                    
                    frame_filename = f"frame_{plate_text}_{timestamp_filename}.jpg"
                    frame_path = os.path.join(result_images_dir, frame_filename)
                    cv2.imwrite(frame_path, display_frame)
                    
                    plate_data = {
                        "number_plate": plate_text,
                        "date_time": timestamp,
                        "camera_id": camera_id,
                        "detector_confidence": float(conf),
                        "ocr_confidence": float(ocr_confidence),
                        "combined_confidence": float(combined_confidence),
                        "plate_image": result_img_filename,
                        "frame_image": frame_filename
                    }
                    
                    if frame_id is not None:
                        plate_data["frame_id"] = frame_id
                    
                    save_to_json(plate_data)
                    plate_tracker.add_plate(plate_text)
                    logger.info(f"Detected: {plate_text} at {timestamp} (Confidence: {combined_confidence:.2f})")
                    
                    frame_results.append({
                        "plate_text": plate_text,
                        "confidence": float(combined_confidence),
                        "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)]
                    })
    
    return display_frame, frame_results

def save_to_json(data):
    """Save data to JSON file, avoiding duplicates based on plate number"""
    try:
        # Load existing data
        try:
            with open(json_file, 'r') as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []

        # Check for duplicates based on 'number_plate' (case-insensitive)
        plate_text = data["number_plate"].upper()
        duplicate_found = False
        for entry in existing_data:
            if entry["number_plate"].upper() == plate_text:
                logger.info(f"Skipping duplicate plate: {plate_text}")
                return  # Skip saving if duplicate found
                # Uncomment below to update duplicates with higher confidence
                # if data["combined_confidence"] > entry["combined_confidence"]:
                #     entry.update(data)
                #     logger.info(f"Updated duplicate plate: {plate_text} with higher confidence")
                # duplicate_found = True
                # break

        # If no duplicate, append the new data
        if not duplicate_found:
            existing_data.append(data)
            logger.info(f"Saved new plate: {plate_text}")

        # Write back to file
        with open(json_file, 'w') as f:
            json.dump(existing_data, f, indent=4)

    except Exception as e:
        logger.error(f"Error saving to JSON: {e}")

def process_video(video_source=0, skip_frames=2, debug=False):
    """Process video from camera or file"""
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        logger.error(f"Error: Could not open video source {video_source}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if isinstance(video_source, str):
        output_path = os.path.join(output_dir, f"processed_{os.path.basename(video_source)}")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    else:
        out = None
    
    frame_count = 0
    processing_times = []
    start_time = time.time()
    
    logger.info(f"Starting real-time processing from video source {video_source}")
    logger.info(f"Saving results to {json_file} and images to {result_images_dir}")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            if isinstance(video_source, str):
                logger.info("End of video file reached")
                break
            else:
                logger.warning("Camera disconnected, attempting to reconnect...")
                time.sleep(1)
                cap = cv2.VideoCapture(video_source)
                continue
        
        frame_count += 1
        
        if frame_count % (skip_frames + 1) == 0:
            process_start = time.time()
            processed_frame, results = process_frame(frame, frame_count, debug)
            process_end = time.time()
            
            processing_time = process_end - process_start
            processing_times.append(processing_time)
            
            if out:
                out.write(processed_frame)
            
            cv2.imshow('Indian License Plate Recognition', processed_frame)
            
            if len(processing_times) >= 10:
                avg_time = sum(processing_times[-10:]) / 10
                actual_fps = 1 / avg_time if avg_time > 0 else 0
                logger.info(f"Processing at {actual_fps:.2f} FPS (avg: {avg_time*1000:.1f}ms)")
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
    
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    logger.info(f"Processing complete. {frame_count} frames processed in {total_time:.2f} seconds")
    logger.info(f"Average processing time: {sum(processing_times)/max(1, len(processing_times)):.3f} seconds per frame")
    logger.info(f"Results saved to {json_file}")

def process_image_folder(folder_path, debug=False):
    """Process all images in a folder"""
    if not os.path.exists(folder_path):
        logger.error(f"Error: Folder {folder_path} not found")
        return
    
    processed_count = 0
    detected_count = 0
    start_time = time.time()
    
    for image_file in sorted(os.listdir(folder_path)):
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
            
        image_path = os.path.join(folder_path, image_file)
        frame = cv2.imread(image_path)
        
        if frame is None:
            logger.error(f"Error: Could not load image {image_path}")
            continue
        
        processed_count += 1
        logger.info(f"Processing image {processed_count}: {image_file}")
        
        processed_frame, results = process_frame(frame, debug=debug)
        detected_count += len(results)
        
        output_path = os.path.join(result_images_dir, f"processed_{image_file}")
        cv2.imwrite(output_path, processed_frame)
        
        cv2.imshow('Indian License Plate Recognition', processed_frame)
        
        key = cv2.waitKey(500)
        if key == 27:
            break
    
    cv2.destroyAllWindows()
    
    end_time = time.time()
    processing_time = end_time - start_time
    logger.info(f"\nProcessing complete. Results saved to {json_file}")
    logger.info(f"Processed {processed_count} images in {processing_time:.2f} seconds")
    logger.info(f"Detected {detected_count} unique license plates")
    logger.info(f"Average processing time: {processing_time/max(1, processed_count):.3f} seconds per image")

def main():
    logger.info("Indian License Plate Recognition System")
    logger.info("--------------------------------------")
    logger.info("1. Process from camera")
    logger.info("2. Process from video file")
    logger.info("3. Process from image folder")
    logger.info("4. Process from camera with debugging")
    choice = input("Enter your choice (1-4): ")
    
    debug_mode = False
    
    if choice == '1':
        try:
            camera_idx = int(input("Enter camera index (default 0): ") or "0")
            skip_frames = int(input("Enter number of frames to skip (default 2): ") or "2")
            process_video(camera_idx, skip_frames)
        except ValueError:
            logger.warning("Invalid input, using default values")
            process_video(0, 2)
    
    elif choice == '2':
        video_path = input("Enter path to video file: ")
        if os.path.exists(video_path):
            skip_frames = int(input("Enter number of frames to skip (default 2): ") or "2")
            process_video(video_path, skip_frames)
        else:
            logger.error(f"Error: File {video_path} not found")
    
    elif choice == '3':
        image_folder = input("Enter path to image folder: ")
        process_image_folder(image_folder)
    
    elif choice == '4':
        try:
            camera_idx = int(input("Enter camera index (default 0): ") or "0")
            skip_frames = int(input("Enter number of frames to skip (default 2): ") or "2")
            process_video(camera_idx, skip_frames, debug=True)
        except ValueError:
            logger.warning("Invalid input, using default values")
            process_video(0, 2, debug=True)
    
    else:
        logger.error("Invalid choice")

if __name__ == "__main__":
    main()