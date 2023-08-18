import cv2
import os

script_dir = os.path.dirname(os.path.abspath(__file__))


xml_file_path = os.path.join(script_dir, 'haarcascade_frontalface_default.xml')
image_path = os.path.join(script_dir, 'test.jpg')

face_cascade = cv2.CascadeClassifier(xml_file_path)

img = cv2.imread(image_path)

if img is None:
    print("Image not loaded or path is incorrect.")
else:
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    
    output_path = os.path.join(script_dir, 'output_image.jpg')
    cv2.imwrite(output_path, img)

    print(f"Image with detected faces saved at: {output_path}")
