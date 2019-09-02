"""
Train a classifier on your own dataset that recognizes people using Transfer Learning.
"""

from keras.models import load_model
import os
import numpy as np
import glob
import argparse
import sys
import cv2
import align
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm

def main(args):

    if args.pretrained_weights is None:
        model_weights = "src/models/facenet_keras_weights.h5"
    else:
        model_weights = args.pretrained_weights

    if args.face_landmark_file:
        face_detector_file = args.face_landmark_file
    else:
        face_detector_file = "src/models/shape_predictor_68_face_landmarks.dat"


    if args.mode == "train":
        train_paths = load_paths(args.trainset_path)
        print("loading weights ..........", time.ctime())
        model = load_model("src/models/facenet_keras.h5")
        model.load_weights(model_weights)
        print("creating embeddings using pretrained weights........", time.ctime())
        embeddings = train_model(train_paths, face_detector_file, model)
        print("training {} classifier......".format(args.classifier2train), time.ctime())
        accuracy = train_classifier(embeddings, args.classifier2train, train_paths, args.classifierdir)
        print("{} model is trained with accuracy of {} ".format(args.classifier2train, accuracy), time.ctime())
        print("model saved in folder {} with name {} ".format(args.classifierdir, args.classifier2train))
    else:
        print("loading weights ........")
        model = load_model("src/models/facenet_keras.h5")
        model.load_weights(model_weights)
        print("predicting ..........")
        if os.path.isdir(args.clfimgpath):
            paths = load_paths(args.clfimgpath)
            prediction = []
            for path in tqdm(paths):
                pred = classify(path, args.clfpath, args.lepath, model, face_detector_file)
                prediction.append((path.split("/")[-1], pred[0]))
        else:
            prediction = classify(args.clfimgpath, args.clfpath, args.lepath, model, face_detector_file)
        print("prediction is : ", prediction)
        return prediction



def align_image(img, face_detector):
    aligndlib = align.AlignDlib(face_detector)
    return aligndlib.align(160, img, aligndlib.getLargestFaceBoundingBox(img),
                               landmarkIndices=align.AlignDlib.OUTER_EYES_AND_NOSE)

def load_paths(trainset_path):
    return np.asarray([os.path.abspath(y) for x in os.walk(trainset_path) for y in glob.glob(os.path.join(x[0], "*.jpg"))])

def load_img(img_path):
    img = cv2.imread(img_path, 1)
    return img[..., ::-1]

def train_model(paths, face_detector, model):
    num_img = len(paths)
    embeddings = np.zeros((num_img, 512))

    for i, path in tqdm(enumerate(paths)):
        img = load_img(path)
        aligned_image = align_image(img, face_detector)
        aligned_image = (aligned_image/255.).astype(np.float32)
        embeddings[i] = model.predict(np.expand_dims(aligned_image, axis=0))[0]
    return embeddings

def classify(img_path, clf_path, le_path, model, face_detector):
    img = load_img(img_path)
    clf = joblib.load(clf_path)
    # encoder = joblib.load(le_path)
    aligned_image = align_image(img, face_detector)
    aligned_image = (aligned_image / 255.).astype(np.float32)
    emb = model.predict([np.expand_dims(aligned_image, axis=0)])[0]
    pred = clf.predict([emb])
    # prediction = encoder.inverse_transform(pred)
    return pred



def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def train_classifier(embedings, clf, trainset_paths, clfpath):
    # encoder = LabelEncoder()
    actual_labels = np.array([os.path.dirname(_).split("/")[-1] for _ in trainset_paths])
    # encoder.fit(actual_labels)
    # y = encoder.transform(actual_labels)
    x_train, x_test, y_train, y_test = train_test_split(embedings, actual_labels, train_size=0.8, shuffle=True)
    if clf == "knn":
        classifier = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    else:
        classifier = joblib.load("src/svm.joblib")   #LinearSVC()

    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    if not os.path.exists(clfpath):
        os.makedirs(clfpath)
    joblib.dump(classifier, os.path.join(clfpath, clf + ".joblib"))
    # joblib.dump(encoder, os.path.join(clfpath, "encoder.joblib"))
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy




def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, choices=['train', 'classify'],
                        help='Indicated new classifier model should be trained or classification model should be used',
                        default='classify')
    parser.add_argument("--trainset_path", type=str, help='Path to trainset data directory')
    parser.add_argument("--pretrained_weights", type=str, help='Path to pretrained weights file')
    parser.add_argument("--testset_path", type=str, help='Path to test data directory')
    parser.add_argument("--face_landmark_file", type=str, help='landmark detection used for dlib')
    parser.add_argument("--classifier2train", type=str, help="which classifier to train", choices=['knn', 'svm'], default='svm')
    parser.add_argument("--classifierdir", type=str, help="where to save trained classifier")
    parser.add_argument("--clfpath", type=str, help="pretrained classifier path")
    parser.add_argument("--lepath", type=str, help="pretrained label encode path for inversetransform prediction")
    parser.add_argument("--clfimgpath", type=str, help="image to predict or classify")
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


