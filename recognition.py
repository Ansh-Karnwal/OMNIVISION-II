import face_recognition
import os, sys
import cv2
import numpy as np
import math
import time
import threading
import queue

# Helper
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


class FaceRecognition:
    def __init__(self):
        # Known faces data
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Results of detection
        self.face_locations = []
        self.face_names = []
        
        # Threading related attributes
        self.frame_queue = queue.Queue(maxsize=1)  # Only store latest frame
        self.results_queue = queue.Queue()
        self.processing_thread = None
        self.is_running = False
        
        # Performance settings
        self.frame_count = 0
        self.frame_skip = 3
        self.detection_model = "hog"  # Use "hog" for speed, "cnn" for accuracy
        self.display_fps = True
        
        # Load known faces
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f"faces/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        print(self.known_face_names)
    
    def process_frames(self):
        """Worker thread function to process frames"""
        while self.is_running:
            try:
                # Get frame from queue with a timeout (so the thread can check is_running periodically)
                frame = self.frame_queue.get(timeout=0.5)
                
                # Process the frame
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                
                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame, model=self.detection_model)
                face_names = []
                
                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    
                    for face_encoding in face_encodings:
                        # See if the face is a match for the known face(s)
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                        name = "Unknown"
                        confidence = '???'
                        
                        # Calculate the shortest distance to face
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        
                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = self.known_face_names[best_match_index]
                                confidence = face_confidence(face_distances[best_match_index])
                        
                        face_names.append(f'{name} ({confidence})')
                
                # Put results in the output queue
                self.results_queue.put((face_locations, face_names))
                
                # Mark task as done
                self.frame_queue.task_done()
                
            except queue.Empty:
                # No frame available, just continue
                continue
            except Exception as e:
                print(f"Error in processing thread: {e}")

    def run_recognition(self):
        # Start the worker thread
        self.is_running = True
        self.processing_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.processing_thread.start()
        
        # Open video capture
        video_capture = cv2.VideoCapture(0)
        
        # Set lower resolution for better performance
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not video_capture.isOpened():
            self.is_running = False
            sys.exit('Video source not found...')
        
        # FPS tracking
        start_time = time.time()
        frame_count = 0
        fps = 0
        
        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            if not ret:
                break
                
            # FPS calculation
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Process only every n-th frame
            self.frame_count += 1
            if self.frame_count % self.frame_skip == 0:
                # If queue is not full, add frame to be processed
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
            
            # Try to get latest results (non-blocking)
            try:
                while not self.results_queue.empty():
                    self.face_locations, self.face_names = self.results_queue.get(block=False)
                    self.results_queue.task_done()
            except queue.Empty:
                pass
            
            # Display the results
            if self.face_locations:
                for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    # Create the frame with the name
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            
            # Display FPS
            if self.display_fps:
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the resulting image
            cv2.imshow('Face Recognition', frame)
            
            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) == ord('q'):
                break
        
        # Clean up
        self.is_running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
            
        video_capture.release()
        cv2.destroyAllWindows()