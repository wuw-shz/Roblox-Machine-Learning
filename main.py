import os
import cv2 as cv
from time import time
from vision import Vision
from pynput import keyboard
from windowcapture import WindowCapture

os.chdir(os.path.dirname(os.path.abspath(__file__)))

wincap = WindowCapture('Roblox')

vision_limestone = Vision(None)

def calculate_confidence(rect):
    (x, y, w, h) = rect
    area = w * h
    max_area = 5000
    return min(area / max_area, 1.0)
key_pressed = None

def on_press(key):
    global key_pressed
    try:
        key_pressed = key.char
    except AttributeError:
        key_pressed = key
listener = keyboard.Listener(on_press=on_press)
listener.start()

loop_time = time()
while(True):
    cascade_limestone = cv.CascadeClassifier('cascade/cascade.xml')
    screenshot = wincap.get_screenshot()
    screenshot_height, screenshot_width, _ = screenshot.shape
    crop_width, crop_height = 150, 150
    center_x = screenshot_width // 2
    center_y = screenshot_height // 2
    x_start = center_x - (crop_width // 2)
    y_start = center_y - (crop_height // 2) +20
    crop_screenshot = screenshot[y_start:y_start + crop_height, x_start:x_start + crop_width]
    rectangles = cascade_limestone.detectMultiScale(screenshot)
    for rect in rectangles:
        confidence = calculate_confidence(rect)
        (x, y, w, h) = rect
        print(confidence)
        confidence_text = f'{confidence:.2f}'
        cv.putText(screenshot, confidence_text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    detection_image = vision_limestone.draw_rectangles(screenshot, rectangles)

    cv.putText(screenshot, f'FPS {1 / (time() - loop_time):.2f}',(0,screenshot_height - 5), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv.imshow('Matches', screenshot)

    loop_time = time()
    cv.waitKey(1)
    if key_pressed == 'q':
        cv.destroyAllWindows()
        break
    elif key_pressed == 'z':
        cv.imwrite(f'saves/{loop_time}.jpg', screenshot)
        print("Saved as positive image.")
    elif key_pressed == 'x':
        cv.imwrite(f'negative/{loop_time}.jpg', screenshot)
        print("Saved as negative image.")

    key_pressed = None

print('Done.')
