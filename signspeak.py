import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, Label, Button, Frame, BOTH, TOP, BOTTOM, LEFT, RIGHT, X, Y, HORIZONTAL
from PIL import Image, ImageTk
import tensorflow as tf
import threading
import queue
import pyttsx3
import os
import time

class SignSpeakApp:
    def __init__(self, root):
        # Initialize the main application window
        self.root = root
        self.root.title("SignSpeak - ASL Recognition")
        self.root.geometry("1200x700")
        
        # Application state variables
        self.is_running = False
        self.is_speaking = False
        self.sequence_mode = False
        self.current_sequence = []
        self.prev_prediction = ""
        self.stable_count = 0
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        
        # Create UI components
        self.create_ui()
        
        # Load the ASL recognition model
        self.load_model()
        
        # Labels for ASL alphabet
        self.labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                       'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del']

    def create_ui(self):
        # Create frame for video feed
        self.video_frame = Frame(self.root, bg="black", width=640, height=480)
        self.video_frame.pack(side=LEFT, padx=10, pady=10, fill=BOTH, expand=True)
        
        # Label for video feed
        self.video_label = Label(self.video_frame)
        self.video_label.pack(fill=BOTH, expand=True)
        
        # Control panel frame
        control_frame = Frame(self.root, bg="#f0f0f0", width=300)
        control_frame.pack(side=RIGHT, padx=10, pady=10, fill=BOTH)
        
        # App title and description
        title_label = Label(control_frame, text="SignSpeak", font=("Arial", 24, "bold"), bg="#f0f0f0")
        title_label.pack(pady=(20, 5))
        
        desc_label = Label(control_frame, text="ASL Recognition System", font=("Arial", 12), bg="#f0f0f0")
        desc_label.pack(pady=(0, 20))
        
        # Prediction display
        pred_frame = Frame(control_frame, bg="#e0e0e0", padx=10, pady=10, relief="ridge", bd=2)
        pred_frame.pack(fill=X, padx=10, pady=10)
        
        Label(pred_frame, text="Current Sign:", font=("Arial", 14), bg="#e0e0e0").pack(anchor="w")
        self.current_sign_label = Label(pred_frame, text="--", font=("Arial", 36, "bold"), bg="#e0e0e0")
        self.current_sign_label.pack(pady=10)
        
        Label(pred_frame, text="Recognized Text:", font=("Arial", 14), bg="#e0e0e0").pack(anchor="w")
        self.text_display = Label(pred_frame, text="", font=("Arial", 18), bg="#e0e0e0", wraplength=250, justify="left")
        self.text_display.pack(pady=10, fill=X)
        
        # Controls
        controls_frame = Frame(control_frame, bg="#f0f0f0", padx=10, pady=10)
        controls_frame.pack(fill=X, padx=10, pady=10)
        
        # Start/Stop button
        self.toggle_button = Button(controls_frame, text="Start Camera", command=self.toggle_camera, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), padx=10, pady=5)
        self.toggle_button.pack(fill=X, pady=5)
        
        # Mode toggle
        mode_frame = Frame(controls_frame, bg="#f0f0f0")
        mode_frame.pack(fill=X, pady=5)
        
        Label(mode_frame, text="Recognition Mode:", bg="#f0f0f0", font=("Arial", 12)).pack(side=LEFT, padx=(0, 10))
        self.mode_button = Button(mode_frame, text="Single Sign", command=self.toggle_mode, bg="#2196F3", fg="white", font=("Arial", 12))
        self.mode_button.pack(side=RIGHT, fill=X, expand=True)
        
        # Text-to-speech button
        self.speak_button = Button(controls_frame, text="Speak Text", command=self.speak_text, bg="#FF9800", fg="white", font=("Arial", 12), state="disabled", padx=10, pady=5)
        self.speak_button.pack(fill=X, pady=5)
        
        # Clear text button
        self.clear_button = Button(controls_frame, text="Clear Text", command=self.clear_text, bg="#f44336", fg="white", font=("Arial", 12), padx=10, pady=5)
        self.clear_button.pack(fill=X, pady=5)
        
        # Status bar
        self.status_bar = Label(self.root, text="Ready", bd=1, relief="sunken", anchor="w", padx=10)
        self.status_bar.pack(side=BOTTOM, fill=X)

    def load_model(self):
        # Try to load a pre-trained model if it exists, otherwise create a placeholder
        model_path = "asl_model.h5"
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            self.update_status("Model loaded successfully")
        else:
            # Create a placeholder CNN model for ASL recognition
            self.model = self.create_model()
            self.update_status("Created new model (not trained)")
    
    def create_model(self):
        # Create a simple CNN model for ASL recognition
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.labels), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def toggle_camera(self):
        if self.is_running:
            self.is_running = False
            self.toggle_button.config(text="Start Camera", bg="#4CAF50")
            self.speak_button.config(state="normal")
            self.update_status("Camera stopped")
        else:
            self.is_running = True
            self.toggle_button.config(text="Stop Camera", bg="#f44336")
            self.speak_button.config(state="disabled")
            self.update_status("Camera started")
            threading.Thread(target=self.video_capture, daemon=True).start()
            threading.Thread(target=self.process_frames, daemon=True).start()

    def toggle_mode(self):
        self.sequence_mode = not self.sequence_mode
        if self.sequence_mode:
            self.mode_button.config(text="Sequence Mode", bg="#9C27B0")
            self.update_status("Switched to sequence mode")
        else:
            self.mode_button.config(text="Single Sign", bg="#2196F3")
            self.update_status("Switched to single sign mode")

    def video_capture(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            self.update_status("Error: Could not open webcam")
            self.is_running = False
            self.toggle_button.config(text="Start Camera", bg="#4CAF50")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Flip horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            
            # Add a region of interest (ROI) rectangle
            roi_size = 224
            x1, y1 = int((frame.shape[1] - roi_size) / 2), int((frame.shape[0] - roi_size) / 2)
            x2, y2 = x1 + roi_size, y1 + roi_size
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Extract the ROI for processing
            roi = frame[y1:y2, x1:x2]
            
            try:
                if not self.frame_queue.full():
                    self.frame_queue.put((frame, roi))
            except:
                pass
            
            # Display the frame
            self.display_frame(frame)
            
            # Introduce a small delay
            time.sleep(0.03)
        
        cap.release()

    def display_frame(self, frame):
        # Convert the OpenCV frame to a tkinter-compatible image
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update the video label
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def preprocess_frame(self, roi):
        # Preprocess the ROI for the model
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply thresholding
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Resize to match model input size
            resized = cv2.resize(thresh, (64, 64))
            
            # Normalize
            normalized = resized / 255.0
            
            # Expand dimensions to create batch of size 1
            preprocessed = np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)
            
            return preprocessed
        except Exception as e:
            print(f"Error preprocessing frame: {e}")
            return None

    def process_frames(self):
        while self.is_running:
            if not self.frame_queue.empty():
                frame, roi = self.frame_queue.get()
                
                # Preprocess the ROI for the model
                preprocessed = self.preprocess_frame(roi)
                
                if preprocessed is not None:
                    # Make a prediction
                    try:
                        prediction = self.model.predict(preprocessed, verbose=0)
                        predicted_class = np.argmax(prediction[0])
                        confidence = prediction[0][predicted_class]
                        
                        # Only consider predictions with confidence above threshold
                        if confidence > 0.7:
                            predicted_sign = self.labels[predicted_class]
                            
                            # Update UI with the prediction
                            self.root.after(0, lambda p=predicted_sign, c=confidence: self.update_prediction(p, c))
                    except Exception as e:
                        print(f"Error making prediction: {e}")
            
            time.sleep(0.1)

    def update_prediction(self, sign, confidence):
        # Update the current sign label
        self.current_sign_label.config(text=sign)
        
        # Handle different modes
        if self.sequence_mode:
            # In sequence mode, we accumulate signs into words
            if sign == self.prev_prediction:
                self.stable_count += 1
                
                # Add to sequence if stable for 10 frames
                if self.stable_count == 10:
                    if sign == 'space':
                        self.current_sequence.append(' ')
                    elif sign == 'del':
                        if self.current_sequence:
                            self.current_sequence.pop()
                    else:
                        self.current_sequence.append(sign)
                    
                    # Update the text display
                    text = ''.join(self.current_sequence)
                    self.text_display.config(text=text)
            else:
                self.stable_count = 0
                
            self.prev_prediction = sign
        else:
            # In single sign mode, we just display the current sign
            if sign == 'space':
                text = self.text_display.cget("text") + " "
            elif sign == 'del':
                text = self.text_display.cget("text")[:-1]
            else:
                text = self.text_display.cget("text") + sign
                
            self.text_display.config(text=text)
        
        # Update status bar with confidence
        self.update_status(f"Detected: {sign} (Confidence: {confidence:.2f})")

    def speak_text(self):
        # Use text-to-speech to vocalize the interpreted text
        if not self.is_speaking:
            text = self.text_display.cget("text")
            if text:
                self.is_speaking = True
                self.speak_button.config(state="disabled")
                
                # Run TTS in a separate thread to avoid UI freezing
                threading.Thread(target=self.tts_thread, args=(text,), daemon=True).start()
            else:
                self.update_status("No text to speak")

    def tts_thread(self, text):
        # Text-to-speech in a separate thread
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            self.update_status(f"TTS Error: {e}")
        finally:
            # Re-enable the speak button
            self.root.after(0, self.enable_speak_button)

    def enable_speak_button(self):
        self.is_speaking = False
        self.speak_button.config(state="normal")

    def clear_text(self):
        # Clear the recognized text
        self.text_display.config(text="")
        self.current_sequence = []
        self.update_status("Text cleared")

    def update_status(self, message):
        # Update the status bar
        self.status_bar.config(text=message)

def main():
    root = tk.Tk()
    app = SignSpeakApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 