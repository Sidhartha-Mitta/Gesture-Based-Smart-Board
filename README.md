# Gesture Based Smart-Board 🖐️🧠

A real-time virtual Smart-Board application controlled entirely through **hand gestures** using a webcam. This intuitive tool allows you to **draw, erase, change slides, record screen/audio, save notes as PDF, and even quit the application** — all with simple finger gestures.

---

## Features
 **Hand Gesture Recognition** using MediaPipe for finger tracking.  
 **Drawing Mode**: Draw on the board by extending the index finger.  
 **Erase Mode**: Erase with thumb and index finger pinched.  
 **Partial Erase Mode**: Erase larger areas when all fingers are extended.  
 **Multi-Slide Support**: Navigate through multiple slides, add or delete slides.  
 **Color Selection**: Select pen color by hovering over color options.  
 **Clear Slide**: Clear the current slide's content.  
 **Save Slides as PDF**: Export all slides into a single PDF file.  
 **Screen and Audio Rec ording**: Record your session with video and microphone audio.  
 **On-Screen Buttons** for navigation and controls.  

---

## Technology Stack
 **OpenCV**: Video capture, drawing, and video recording.  
 **MediaPipe**: Hand landmark detection and gesture recognition.  
 **NumPy**: Image processing and handling coordinates.  
 **FPDF**: Exporting slides to PDF.  
 **PyAudio**: Audio recording.  
 **Wave**: Saving audio files.  
 **Threading**: For concurrent audio recording.  

---

## ✋ Gesture Controls

| Gesture                               | Action                    |
|---------------------------------------|---------------------------|
| 👉 Index finger extended              | Draw on the board         |
| 🤏 Thumb + Index close                | Erase small portions      |
| 🖐️ All fingers extended and swiped    | Erase larger areas        |
| 👇 Hover over colored squares         | Change pen color          |
| ⏭️ Hover over NEXT / PREV             | Navigate slides           |
| 🗑️ Hover over CLEAR / DEL             | Clear or delete slide     |
| 💾 Hover over SAVE                    | Save all slides as PDF    |
| 🎙️ Hover over REC / STOP              | Start/Stop recording      |
| 👍 press 'q'                          | Quit the application      |


---

## 🛠️ Installation

Install required packages with:

```bash
pip install opencv-python mediapipe numpy fpdf pyaudio
```

---

## ▶️ Start the application
```bash
python SmartBoard.py
```





