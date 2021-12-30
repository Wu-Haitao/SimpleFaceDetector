import cv2

trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
webCam = cv2.VideoCapture(0)
while True:
    isSuccessful, img = webCam.read()
    if not isSuccessful:
        break
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_data.detectMultiScale(grayscale_img)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow('Face Detector', img)
    key = cv2.waitKey(1)
    if key == 32:
        break
webCam.release()
