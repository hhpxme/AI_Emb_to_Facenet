import cv2
import os


def cut_frame_video(vid, folder_name):
    path = os.path.join('data/', folder_name)
    os.mkdir(path)
    cam = cv2.VideoCapture(vid)
    currentframe = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        if ret:
            name = 'data/' + folder_name + '/image_frame' + str(currentframe) + '.jpg'

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break

    cam.release()
    print('Capture video success')

