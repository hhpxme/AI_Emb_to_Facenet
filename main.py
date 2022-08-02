import detection
import face_embbeding
import preprocessing

import tensorflow as tf
from numpy import savez_compressed
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle


video_path = 'video_data/owner_1.mp4'
data_training = 'data/'
# load the facenet model
facenet_model = tf.keras.models.load_model('models/facenet/facenet_keras.h5')

preprocessing.cut_frame_video(video_path, 'owner_1')

trainX, trainY = detection.load_dataset(data_training)

trainX = face_embbeding.embedding(facenet_model, trainX)
# save arrays to one file in compressed format
savez_compressed('numpy_data/data_transfer_embeddings.npz', trainX, trainY)


out_encoder = LabelEncoder()
out_encoder.fit(trainY)
trainY = out_encoder.transform(trainY)
pickle_labels = 'models/emb/out_encoder.pkl'
with open(pickle_labels, 'wb') as file:
    pickle.dump(out_encoder, file)

# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainY)
pickle_model = 'models/emb/faces_svm.pkl'
with open(pickle_model, 'wb') as file:
    pickle.dump(model, file)
