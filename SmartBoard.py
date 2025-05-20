import cv2
import numpy as np
import mediapipe as mp
from fpdf import FPDF
import pyaudio
import wave
import threading

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Drawing Parameters
pen_colors = {
    1: (0, 0, 255),  # Red
    2: (0, 255, 0),  # Green
    3: (255, 0, 0),  # Blue
    4: (0, 255, 255),  # Yellow
    5: (255, 0, 255),  # Magenta
}
pen_color = pen_colors[1]  # Default color
pen_thickness = 5

# Multi-slide support
slides = [np.zeros((720, 1280, 3), dtype=np.uint8)]
current_slide = 0

# Open Camera to capute our picture
cap = cv2.VideoCapture(0)

# Set frame width and height to a larger size so that our frame is fixed with this dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# intializing all buttons with false intially and become true when it get activate
next_button_active = False
prev_button_active = False
del_button_active = False

# Buffer to store recent points for smoother drawing
points_buffer = []

# Screen and Voice Recording Variables
is_recording = False
screen_recorder = None
voice_recorder = None
audio_frames = []

# Audio Recording Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
# is_all_fingers_extented is a function returns true when all fingures are opened....
def is_all_fingers_extended(hand_landmarks):
    return all(
        hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y
        for tip in [8, 12, 16, 20]
    )
'''If we want to open only index finger 8.y < 6.y and rest should be greater is_index_finger_extended return true
if index finger is opened'''
def is_index_finger_extended(hand_landmarks):
    return (
        hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and
        hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and
        hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and
        hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
    )

def is_index_thumb_close(hand_landmarks):
    index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
    thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
    return np.linalg.norm(index_tip - thumb_tip) < 0.05


def is_near_clear_button(x, y):
    return 300 <= x <= 400 and 10 <= y <= 60

def is_near_next_button(x, y):
    return 410 <= x <= 510 and 10 <= y <= 60

def is_near_prev_button(x, y):
    return 520 <= x <= 620 and 10 <= y <= 60

def is_near_save_button(x, y):
    return 630 <= x <= 730 and 10 <= y <= 60

def is_near_delete_button(x, y):
    return 740 <= x <= 840 and 10 <= y <= 60

def is_near_record_button(x, y):
    return 850 <= x <= 950 and 10 <= y <= 60

def is_near_stop_button(x, y):
    return 960 <= x <= 1060 and 10 <= y <= 60

def get_selected_color(x, y):
    if 10 <= y <= 60 and 10 <= x <= 310:
        return (x - 10) // 60 + 1
    return None

def save_pdf():
    pdf = FPDF()
    for i, slide in enumerate(slides):
        filename = f"slide_{i+1}.jpg"
        cv2.imwrite(filename, slide)
        pdf.add_page()
        pdf.image(filename, x=0, y=0, w=210, h=297)  # A4 size
    pdf.output("Virtual_Board.pdf")
    print("PDF Saved as Virtual_Board.pdf")

def add_new_slide():
    """Add a new blank slide and set it as the current slide."""
    slides.append(np.zeros((720, 1280, 3), dtype=np.uint8))
    return len(slides) - 1  # Return the index of the new slide

def delete_current_slide():
    """Delete the current slide and switch to the previous slide."""
    global current_slide
    if len(slides) > 1:
        slides.pop(current_slide)
        if current_slide >= len(slides):
            current_slide = len(slides) - 1
    else:
        # If it's the last slide, reset it to a blank slide
        slides[current_slide][:] = (0, 0, 0)

def start_recording():
    global is_recording, screen_recorder, voice_recorder, audio_frames
    is_recording = True
    screen_recorder = cv2.VideoWriter('screen_recording.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (1280, 720))
    audio_frames = []
    voice_recorder = pyaudio.PyAudio()
    stream = voice_recorder.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    threading.Thread(target=record_audio, args=(stream,)).start()

def record_audio(stream):
    while is_recording:
        data = stream.read(CHUNK)
        audio_frames.append(data)

def stop_recording():
    global is_recording, screen_recorder, voice_recorder, audio_frames
    if is_recording:
        is_recording = False
        screen_recorder.release()
        voice_recorder.terminate()
        wf = wave.open('voice_recording.wav', 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(voice_recorder.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(audio_frames))
        wf.close()
        print("Recording stopped and saved.")
        print("Audio saved as voice_recording.wav")
        print("Video saved as screen_recording.avi")

prev_x, prev_y = None, None
mode = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger = hand_landmarks.landmark[8]
            x, y = int(index_finger.x * frame.shape[1]), int(index_finger.y * frame.shape[0])

            selected_color = get_selected_color(x, y)
            if selected_color and selected_color in pen_colors:
                pen_color = pen_colors[selected_color]

            if is_near_clear_button(x, y):
                slides[current_slide][:] = (0, 0, 0)  # Clear current slide
                points_buffer = []  # Clear the points buffer
            elif is_near_next_button(x, y):
                if not next_button_active:
                    if current_slide == len(slides) - 1:
                        # If on the last slide, add a new slide
                        current_slide = add_new_slide()
                    else:
                        current_slide += 1  # Move to the next slide
                    next_button_active = True
            elif is_near_prev_button(x, y):
                if not prev_button_active:
                    if current_slide > 0:
                        current_slide -= 1  # Move to the previous slide
                    prev_button_active = True
            elif is_near_save_button(x, y):
                save_pdf()
            elif is_near_delete_button(x, y):
                if not del_button_active:
                    delete_current_slide()
                    del_button_active = True
            elif is_near_record_button(x, y):
                if not is_recording:
                    start_recording()
            elif is_near_stop_button(x, y):
                if is_recording:
                    stop_recording()
            elif is_all_fingers_extended(hand_landmarks):
                mode = "erase_partial"
            elif is_index_finger_extended(hand_landmarks):
                mode = "write"
            elif is_index_thumb_close(hand_landmarks):
                mode = "erase"
            else:
                mode = None

            # Reset button active flags if hand moves away
            if not is_near_next_button(x, y):
                next_button_active = False
            if not is_near_prev_button(x, y):
                prev_button_active = False
            if not is_near_delete_button(x, y):
                del_button_active = False

            if mode == "write":
                points_buffer.append((x, y))

                # Apply moving average for smoothing
                if len(points_buffer) > 2:
                    smoothed_points = []
                    for i in range(1, len(points_buffer) - 1):
                        avg_x = (points_buffer[i-1][0] + points_buffer[i][0] + points_buffer[i+1][0]) // 3
                        avg_y = (points_buffer[i-1][1] + points_buffer[i][1] + points_buffer[i+1][1]) // 3
                        smoothed_points.append((avg_x, avg_y))

                    # Draw a polyline for a smooth stroke
                    if len(smoothed_points) > 1:
                        cv2.polylines(slides[current_slide], [np.array(smoothed_points, np.int32)], False, pen_color, pen_thickness, cv2.LINE_AA)

                # Keep buffer size within limit to avoid lag
                if len(points_buffer) > 15:
                    points_buffer.pop(0)

            elif mode == "erase":
                cv2.circle(slides[current_slide], (x, y), 20, (0, 0, 0), -1)
                points_buffer = []  # Clear the points buffer
            elif mode == "erase_partial":
                cv2.circle(slides[current_slide], (x, y), 40, (0, 0, 0), -1)
                points_buffer = []  # Clear the points buffer
            else:
                points_buffer = []  # Clear the points buffer

    # Display color menu and buttons
    color_menu_width = min(1070, frame.shape[1] - 20)  # setting the frame
    color_menu = np.zeros((50, color_menu_width, 3), dtype=np.uint8)
    for i, color in pen_colors.items():
        cv2.rectangle(color_menu, ((i-1)*60, 0), (i*60, 50), color, -1)
    cv2.rectangle(color_menu, (300, 0), (400, 50), (255, 255, 255), -1)
    cv2.putText(color_menu, "CLEAR", (310, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.rectangle(color_menu, (410, 0), (510, 50), (255, 255, 255), -1)
    cv2.putText(color_menu, "NEXT", (420, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.rectangle(color_menu, (520, 0), (620, 50), (255, 255, 255), -1)
    cv2.putText(color_menu, "PREV", (530, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.rectangle(color_menu, (630, 0), (730, 50), (255, 255, 255), -1)
    cv2.putText(color_menu, "SAVE", (640, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.rectangle(color_menu, (740, 0), (840, 50), (255, 255, 255), -1)
    cv2.putText(color_menu, "DEL", (750, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.rectangle(color_menu, (850, 0), (950, 50), (255, 255, 255), -1)
    cv2.putText(color_menu, "REC", (860, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.rectangle(color_menu, (960, 0), (1060, 50), (255, 255, 255), -1)
    cv2.putText(color_menu, "STOP", (970, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # adjusting color frames
    frame[10:60, 10:10 + color_menu_width] = color_menu

    # Display mode and slide info
    mode_text = "Writing" if mode == "write" else "Erasing" if mode == "erase" else "Partial Erase" if mode == "erase_partial" else "None"
    cv2.putText(frame, f"Mode: {mode_text}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, f"Slide: {current_slide + 1}/{len(slides)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Merge slides with frame
    slides[current_slide] = cv2.resize(slides[current_slide], (frame.shape[1], frame.shape[0]))
    frame = cv2.addWeighted(frame, 0.5, slides[current_slide], 0.5, 0)

    # Record screen if recording is active
    if is_recording:
        screen_recorder.write(frame)

    cv2.imshow("Virtual Smartboard", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):  # New slide
        current_slide = add_new_slide()
    elif key == ord('d'):  # Download PDF
        save_pdf()

cap.release()
cv2.destroyAllWindows()

