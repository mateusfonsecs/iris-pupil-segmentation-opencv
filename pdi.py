import cv2
import numpy as np

def detect_eyes(image):
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=15)
    return eyes

def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
        eyes = detect_eyes(gray)

        for (ex, ey, ew, eh) in eyes:
            largest_contour = None
            max_ratio = 0.0
            roi_gray = gray[ey:ey+eh, ex:ex+ew]

            roi_gray_filtered = cv2.bilateralFilter(roi_gray, d=9, sigmaColor=75, sigmaSpace=75)

            threshold = cv2.adaptiveThreshold(roi_gray_filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)

            threshold = cv2.medianBlur(threshold, 9)

            kernel = np.ones((2, 1), np.uint8)
            threshold = cv2.erode(threshold, kernel, iterations=1)

            threshold = cv2.bitwise_not(threshold)
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                try:
                    ratio = area / perimeter
                except ZeroDivisionError:
                    ratio = 0

                if ratio > max_ratio and area > (10) and ratio > 0.2:
                    max_ratio = ratio
                    largest_contour = contour 

            cor = (0, 0, 255)
            if largest_contour is not None:
                cv2.drawContours(frame, [largest_contour + (ex, ey)], 0, cor, -1)

        frame_filter = cv2.bilateralFilter(frame, d=12, sigmaColor=15, sigmaSpace=15)
        # frame_s = cv2.convertScaleAbs(gray, alpha=3, beta=10)

        cv2.imshow("Pupila com Filtro", frame_filter)
        # cv2.imshow("Imagem S", frame_s)
        cv2.imshow("Imagem Original", frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
