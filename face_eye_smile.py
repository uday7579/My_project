import cv2

# PROJECT FACE , EYE ,SMILE DETECTION

face_cascade = cv2.CascadeClassifier("opencv_projects/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("opencv_projects/haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("opencv_projects/haarcascade_smile.xml")

cap  = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        reason_of_face_gray = gray[y:y + h, x:x + w]
        reason_of_face_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(reason_of_face_gray, 1.1, 10)
        if len(eyes) > 0:
            cv2.putText(frame, "Eyes Detected", (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

        smiles = smile_cascade.detectMultiScale(reason_of_face_gray, 1.7, 20)
        if len(smiles) > 0:
            cv2.putText(frame, "Smile Detected", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Smart Face Detector", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
