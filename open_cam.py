import cv2
import matplotlib.pyplot as plt

vid = cv2.VideoCapture(0)

image = []

while (True):
    ret, frame = vid.read()
    image.append(frame)
    if len(image) == 6:
        break

for i in range(len(image)):
    plt.subplot(2, 3, i + 1)
    plt.axis('off')
    plt.imshow(image[i])

plt.show()
vid.release()
