# calculate a face embedding for each face in the dataset using facenet
from numpy import expand_dims
from numpy import asarray


# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


def embedding(model, trainX):
    # convert each face in the train set to an embedding
    newTrainX = list()
    for face_pixels in trainX:
        emb = get_embedding(model, face_pixels)
        newTrainX.append(emb)
    newTrainX = asarray(newTrainX)

    return newTrainX

