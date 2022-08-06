import tensorflow as tf
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
import pickle
import cv2
import face_embbeding


vid = cv2.VideoCapture(0)

img_path = 'beauty_20220724113920.jpg'
img = cv2.imread(img_path)
detector = MTCNN()
img_size = (160, 160)

facenet_model = tf.keras.models.load_model('models/facenet/facenet_keras.h5')

# Load SVM model từ file
pkl_filename = 'models/emb/faces_svm.pkl'
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

# Load output_enc từ file để hiển thị nhãn
pkl_filename = 'models/emb/out_encoder.pkl'
with open(pkl_filename, 'rb') as file:
    output_enc = pickle.load(file)


while (True):
    ret, frame = vid.read()
    image = frame

    pixels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = detector.detect_faces(pixels)

    if len(results) > 0:
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(img_size)

        face_emb = face_embbeding.get_embedding(facenet_model, np.array(image))

        face_emb = np.expand_dims(face_emb, axis=0)

        y_hat = pickle_model.predict(face_emb)
        y_hat_prob = pickle_model.predict_proba(face_emb)

        predict_names = output_enc.inverse_transform(y_hat)
        predict_prob = y_hat_prob[0, y_hat[0]] * 100

        if predict_names != None:
            cv2.putText(img, predict_names[0], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            print('Predict: ' + predict_names[0])
            print('Expected: %.3f' % predict_prob)


cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
