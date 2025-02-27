import cv2
import time
from ultralytics import YOLO
import threading
import queue

def detect_objects_realtime(camera_id=0, conf_threshold=0.5):
    """
    Real-time weapons and knives detection using a webcam or camera feed.
    
    Parameters:
    camera_id (int): Camera device ID (0 for default webcam)
    conf_threshold (float): Confidence threshold for detections (0.0-1.0)
    """
    # Load the YOLO model
    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')
    
    # Open the webcam
    cap = cv2.VideoCapture(camera_id)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Variables for FPS calculation
    fps_start_time = 0
    fps = 0
    frame_count = 0
    
    print("Real-time detection started. Press 'q' to quit.")
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # FPS calculation
        frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_start_time = time.time()
        
        # Perform detection on the frame
        results = yolo_model(frame)
        
        # Process results and draw on the frame
        detections_count = 0
        for result in results:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy
            
            for pos, detection in enumerate(detections):
                if conf[pos] >= conf_threshold:
                    detections_count += 1
                    xmin, ymin, xmax, ymax = detection
                    label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}"
                    
                    color = (0, int(cls[pos]) * 85, 255)  # Different color per class
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        # Display status information
        cv2.putText(frame, f"FPS: {fps}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Threshold: {conf_threshold:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Detections: {detections_count}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Display the frame with detections
        cv2.imshow("Real-time Weapons & Knives Detection", frame)
        
        # Check for user input to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            conf_threshold = min(conf_threshold + 0.05, 1.0)
            print(f"Confidence threshold increased to {conf_threshold:.2f}")
        elif key == ord('-') or key == ord('_'):
            conf_threshold = max(conf_threshold - 0.05, 0.05)
            print(f"Confidence threshold decreased to {conf_threshold:.2f}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")


# For better performance with multi-threading
class ThreadedDetector:
    """
    Multi-threaded implementation for improved performance in real-time detection.
    Separates capture, detection, and display processes into different threads.
    """
    import threading
    import queue
    
    def __init__(self, model_path='./best.pt', camera_id=0, conf_threshold=0.5):
        self.model_path = model_path
        self.camera_id = camera_id
        self.conf_threshold = conf_threshold
        
        # Initialize queues for thread communication
        self.frame_queue = queue.Queue(maxsize=2)
        self.results_queue = queue.Queue(maxsize=2)
        
        # Control flags
        self.is_running = False
        
        # FPS tracking
        self.fps = 0
        self.detections_count = 0
        
    def start(self):
        """Start the real-time detection process"""
        if self.is_running:
            print("Detection is already running.")
            return
        
        # Load the YOLO model
        self.model = YOLO(self.model_path)
        
        # Open the webcam
        self.cap = cv2.VideoCapture(self.camera_id)
        
        # Check if the webcam is opened correctly
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}.")
            return
        
        # Start threads
        self.is_running = True
        
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.detection_thread = threading.Thread(target=self._process_frames)
        self.display_thread = threading.Thread(target=self._display_results)
        
        self.capture_thread.daemon = True
        self.detection_thread.daemon = True
        self.display_thread.daemon = True
        
        self.capture_thread.start()
        self.detection_thread.start()
        self.display_thread.start()
        
        print("Real-time detection started.")
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press '+' to increase confidence threshold")
        print("  - Press '-' to decrease confidence threshold")
        
        # Wait for display thread to finish (when user quits)
        self.display_thread.join()
        self._cleanup()
        
    def _capture_frames(self):
        """Thread function to capture frames from camera"""
        prev_time = 0
        
        while self.is_running:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Error: Failed to capture frame.")
                self.is_running = False
                break
            
            # Calculate capture FPS
            curr_time = time.time()
            if prev_time > 0:
                self.fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            # Add frame to queue, skip if queue is full (to avoid delay)
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                pass
    
    def _process_frames(self):
        """Thread function to process frames with YOLO model"""
        while self.is_running:
            try:
                # Get a frame from the queue
                frame = self.frame_queue.get(timeout=1)
                
                # Process with YOLO
                results = self.model(frame)
                
                # Put results in the output queue
                try:
                    self.results_queue.put((frame, results), block=False)
                except queue.Full:
                    pass
                
                self.frame_queue.task_done()
            except queue.Empty:
                continue
    
    def _display_results(self):
        """Thread function to display detection results"""
        while self.is_running:
            try:
                # Get processed frame and results
                frame, results = self.results_queue.get(timeout=1)
                
                # Process results and draw on the frame
                self.detections_count = 0
                for result in results:
                    classes = result.names
                    cls = result.boxes.cls
                    conf = result.boxes.conf
                    detections = result.boxes.xyxy
                    
                    for pos, detection in enumerate(detections):
                        if conf[pos] >= self.conf_threshold:
                            self.detections_count += 1
                            xmin, ymin, xmax, ymax = detection
                            label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}"
                            
                            color = (0, int(cls[pos]) * 85, 255)  # Different color for each class
                            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                            cv2.putText(frame, label, (int(xmin), int(ymin) - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                
                # Display status information on the frame
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Threshold: {self.conf_threshold:.2f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Detections: {self.detections_count}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                
                # Display the frame with detections
                cv2.imshow("Real-time Object Detection", frame)
                
                # Check for user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.is_running = False
                    break
                elif key == ord('+') or key == ord('='):
                    self.conf_threshold = min(self.conf_threshold + 0.05, 1.0)
                    print(f"Confidence threshold increased to {self.conf_threshold:.2f}")
                elif key == ord('-') or key == ord('_'):
                    self.conf_threshold = max(self.conf_threshold - 0.05, 0.05)
                    print(f"Confidence threshold decreased to {self.conf_threshold:.2f}")
                
                self.results_queue.task_done()
            except queue.Empty:
                continue
    
    def _cleanup(self):
        """Clean up resources when detection is stopped"""
        self.is_running = False
        
        # Release resources
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("Detection stopped.")

# Example usage:
if __name__ == "__main__":
    # Option 1: Simple real-time detection
    # detect_objects_realtime()
    
    # Option 2: Multi-threaded detection for better performance
    detector = ThreadedDetector()
    detector.start()