## SignSpeak - ASL Recognition Application

SignSpeak is a desktop application that uses computer vision and deep learning to recognize American Sign Language (ASL) letters and convert them to text in real-time.

## Features Listed Below:


- Real-time ASL letter recognition from webcam input
- Preprocessing of hand gestures for better recognition
- Two recognition modes:
  - Single Sign Mode - displays each recognized sign immediately
  - Sequence Mode - collects stable signs to form words/phrases
- Text-to-speech functionality to vocalize recognized text
- User-friendly GUI with live video feed

## Getting Started

### Prerequisites

- Python 3.7 or later
- Webcam

### Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/signspeak.git
cd signspeak
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Run the application:
```
python signspeak.py
```

## Using the Application

1. Start the application by running `python signspeak.py`
2. Click the "Start Camera" button to begin capturing video from your webcam
3. Position your hand within the green square (region of interest)
4. Perform ASL signs and observe the recognition results
5. Toggle between "Single Sign" and "Sequence Mode" to change recognition behavior
6. Use "Speak Text" button to vocalize the recognized text
7. "Clear Text" button will reset the text display

## Recognition Modes

- **Single Sign Mode**: Each recognized sign is immediately added to the text display.
- **Sequence Mode**: The application waits for a stable sign (same sign detected for multiple consecutive frames) before adding it to the text display.

## Special Signs

- **space**: Inserts a space in the text
- **del**: Deletes the last character from the text

## Training Your Own Model

If you want to train the model on your own dataset:

1. Prepare a dataset with the following structure:
```
dataset/
  A/
    image1.jpg
    image2.jpg
    ...
  B/
    image1.jpg
    ...
  ...
```

2. Run the training script:
```
python model_trainer.py path/to/dataset
```

3. The trained model will be saved as `asl_model.h5` and loaded automatically the next time you run the application.

## Contributing

Contributions are welcome! Feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This application uses TensorFlow for deep learning
- OpenCV for computer vision processing
- Tkinter for the graphical user interface 
