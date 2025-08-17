import cv2
print(cv2.__version__)  # Verifica a versão do OpenCV
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    cv2.imshow('Test', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
cap.release()