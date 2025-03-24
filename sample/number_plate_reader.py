import cv2
import json
import datetime
import os
import torch
import numpy as np
import easyocr
import re
import logging
import time
from PIL import Image, ImageEnhance, ImageFilter

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("anpr_log.txt"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Configuration
CAMERA_ID = "CAM01"
confidence_threshold = 0.5
duplicate_window = 5  # Seconds to consider duplicates

# Indian license plate patterns
indian_plate_patterns = [
    r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$',  # Standard format: MH12AB1234
    r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,1}[0-9]{4}$',  # Older format: MH12A1234
    r'^[A-Z]{3}[0-9]{4}$',                      # Diplomat plates: UN1234
    r'^[A-Z]{2}[0-9]{3,4}$',                    # Older format: MH1234
    r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{3,4}$'     # Newer format: DL01AB1234
]

# Create output directories
result_images_dir = "result_images"
debug_dir = "debug_images"
os.makedirs(result_images_dir, exist_ok=True)
os.makedirs(debug_dir, exist_ok=True)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# Load YOLOv5 model
def load_model():
    try:
        model = torch.hub.load('./yolov5', 'custom', 
                             path='./yolov5/runs/train/exp/weights/best.pt', 
                             source='local', force_reload=True)
        model.conf = confidence_threshold
        
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

# Load the YOLOv5 model
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
        
        return processed_images
    
    @staticmethod
    def correct_plate_text(text):
        """Correct common OCR mistakes in Indian license plates"""
        corrections = {
            '0': 'O', 'Q': 'O', 'D': '0',  # Common confusions
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
                    # If it's a digit but should be a letter
                    if text[i] == '0':
                        text = text[:i] + 'O' + text[i+1:]
                    elif text[i] == '1':
                        text = text[:i] + 'I' + text[i+1:]
                    elif text[i] == '8':
                        text = text[:i] + 'B' + text[i+1:]
            
            # Last four characters are typically digits
            for i in range(max(0, len(text) - 4), len(text)):
                if text[i].isalpha():
                    # If it's a letter but should be a digit
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
        
        # Save debug images if requested
        if debug:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_image_path = os.path.join(debug_dir, f"debug_{timestamp}.jpg")
            
            # Create a grid of all processed images
            h, w = processed_images[0].shape
            grid = np.zeros((h * 2, w * 3), dtype=np.uint8)
            
            for i, img in enumerate(processed_images[:6]):  # Up to 6 images in a 2x3 grid
                r, c = i // 3, i % 3
                grid[r*h:(r+1)*h, c*w:(c+1)*w] = img
                
            cv2.imwrite(debug_image_path, grid)
        
        # Try OCR on each processed image
        for idx, img in enumerate(processed_images):
            # EasyOCR requires RGB images
            if len(img.shape) == 2:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = img
                
            # Run EasyOCR
            results = reader.readtext(img_rgb, detail=1, paragraph=False)
            
            # Process results
            for (bbox, text, prob) in results:
                if prob > 0.3:  # Filter by confidence
                    # Clean and correct the text
                    text = LicensePlateProcessor.correct_plate_text(text)
                    
                    # Only keep reasonable length results
                    if 4 <= len(text) <= 10:
                        all_results.append((text, prob))
        
        # If no results, try direct OCR with lower confidence
        if not all_results:
            results = reader.readtext(plate_img, detail=1, paragraph=False, 
                                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            for (bbox, text, prob) in results:
                if prob > 0.2:  # Lower threshold for direct attempt
                    text = LicensePlateProcessor.correct_plate_text(text)
                    if 4 <= len(text) <= 10:
                        all_results.append((text, prob))
        
        # If still no results
        if not all_results:
            return "", 0.0
        
        # Sort by confidence
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        # Create a dictionary to count occurrences with confidence weighting
        weighted_counts = {}
        for text, prob in all_results:
            if text in weighted_counts:
                weighted_counts[text] += prob
            else:
                weighted_counts[text] = prob
        
        # Get the most likely result
        if weighted_counts:
            most_likely = max(weighted_counts.items(), key=lambda x: x[1])
            return most_likely[0], float(most_likely[1])
        
        return "", 0.0
    
    @staticmethod
    def validate_plate(plate_text):
        """Validate if the plate text is likely to be a valid Indian license plate"""
        # Check if empty
        if not plate_text:
            return False
        
        # Check for minimum length
        if len(plate_text) < 4 or len(plate_text) > 10:
            return False
        
        # Check against Indian license plate patterns
        for pattern in indian_plate_patterns:
            if re.match(pattern, plate_text):
                return True
        
        # If no exact match, check if it's close to a valid pattern
        if len(plate_text) >= 4:
            # Check if first two characters are letters (standard in Indian plates)
            if not (plate_text[0].isalpha() and plate_text[1].isalpha()):
                return False
            
            # Check if it has at least one digit
            if not any(char.isdigit() for char in plate_text):
                return False
            
            # More than 50% of characters after the first two should be digits
            if sum(1 for c in plate_text[2:] if c.isdigit()) < len(plate_text[2:]) / 2:
                return False
            
            # If it passes these checks, it might be valid with some OCR errors
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
        
        # Clean old entries
        self.recent_plates = [entry for entry in self.recent_plates 
                             if (current_time - entry["time"]) < self.time_window]
        
        # Check for duplicates
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
    
    # Add padding (but stay within image bounds)
    height, width = frame.shape[:2]
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)
    
    return frame[y_min:y_max, x_min:x_max]

# Modified: Save only required data to JSON
def save_data(number_plate):
    current_time = datetime.datetime.now()
    date = current_time.strftime("%Y-%m-%d")
    time_str = current_time.strftime("%H:%M:%S")
    
    data_entry = {
        "number_plate": number_plate,
        "date": date,
        "time": time_str,
        "camera_id": CAMERA_ID
    }

    json_file = 'number_plate_data.json'
    if os.path.exists(json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
    else:
        data = []

    data.append(data_entry)
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)
    
    logger.info(f"Saved: {number_plate} at {date} {time_str}")

def process_frame(frame, debug=False):
    """Process a single frame and detect license plates"""
    # Make a copy of the frame for drawing
    display_frame = frame.copy()
    
    # YOLOv5 inference
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()
    
    # Sort detections by confidence (highest first)
    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    
    detected_plates = []
    
    for det in detections:
        x_min, y_min, x_max, y_max, conf, cls = det
        if conf > confidence_threshold:
            # Extract plate region with padding
            plate_img = extract_plate_region(frame, det)
            
            if plate_img.size == 0:
                continue
            
            # Try to read the plate with EasyOCR
            plate_text, ocr_confidence = LicensePlateProcessor.read_plate(plate_img, debug)
            
            # Validate the plate
            if plate_text and LicensePlateProcessor.validate_plate(plate_text):
                # Combine detector and OCR confidence
                combined_confidence = (conf + ocr_confidence) / 2
                
                # Draw bounding box and text
                cv2.rectangle(display_frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), 
                            (0, 255, 0), 2)
                
                # Add text above the bounding box
                cv2.putText(display_frame, f"{plate_text} ({combined_confidence:.2f})", 
                           (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           (0, 255, 0), 2)
                
                timestamp_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if not plate_tracker.is_duplicate(plate_text):
                    # Still save the image for debugging
                    plate_img_filename = f"{plate_text}_{timestamp_filename}.jpg"
                    plate_img_path = os.path.join(result_images_dir, plate_img_filename)
                    cv2.imwrite(plate_img_path, plate_img)
                    
                    # Save the frame with bounding box
                    frame_img_filename = f"frame_{plate_text}_{timestamp_filename}.jpg"
                    frame_img_path = os.path.join(result_images_dir, frame_img_filename)
                    cv2.imwrite(frame_img_path, display_frame)
                    
                    # Save only required data to JSON
                    save_data(plate_text)
                    plate_tracker.add_plate(plate_text)
                    
                    detected_plates.append({
                        "plate_text": plate_text,
                        "confidence": float(combined_confidence),
                        "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)]
                    })
    
    return display_frame, detected_plates

def main():
    """Main function to run the license plate recognition system"""
    print("License Plate Recognition System")
    print("1. Process from camera")
    print("2. Process from image file")
    print("3. Enable debug mode (saves preprocessing steps)")
    choice = input("Enter your choice (1-3): ")
    
    debug_mode = False
    if choice == '3':
        debug_mode = True
        choice = input("Debug mode enabled. Now select source (1-2): ")
    
    cap = None
    if choice == '1':
        print("Starting camera. Press 'Enter' to scan, 'q' to quit...")
        cap = cv2.VideoCapture(0)  # Default webcam
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            display_frame = frame.copy()
            cv2.putText(display_frame, "Press ENTER to scan, 'q' to quit", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("License Plate Recognition", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                print("Scanning number plate...")
                processed_frame, detected_plates = process_frame(frame, debug_mode)
                
                if detected_plates:
                    for plate in detected_plates:
                        print(f"Detected Number Plate: {plate['plate_text']}")
                else:
                    print("No number plate detected.")
                
                cv2.imshow("License Plate Recognition", processed_frame)
                cv2.waitKey(2000)  # Display result for 2 seconds
            
            elif key == ord('q'):  # Quit on 'q'
                break
    
    elif choice == '2':
        image_path = input("Enter image path: ")
        if not os.path.exists(image_path):
            print(f"Error: Image {image_path} not found.")
            return
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}.")
            return
        
        print("Processing image...")
        processed_frame, detected_plates = process_frame(frame, debug_mode)
        
        if detected_plates:
            for plate in detected_plates:
                print(f"Detected Number Plate: {plate['plate_text']}")
        else:
            print("No number plate detected.")
        
        cv2.imshow("License Plate Recognition", processed_frame)
        cv2.waitKey(0)  # Wait for any key press
    
    else:
        print("Invalid choice.")
    
    # Clean up
    if cap:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize JSON file if not exists
    json_file = 'number_plate_data.json'
    if not os.path.exists(json_file):
        with open(json_file, 'w') as file:
            json.dump([], file)

    main()