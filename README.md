# FaceRecognition
Creating face recognition system

System using pretrained open face model on custome datasets.

Detection of face from images are done using harcascade classifier and another from Multitasking cascading CNN(MTCNN) which have more accuract to detact face but slow as uses NN to predict.

Encodings from openface model trained on VGGface2 datasets are used and then classifier is created to predict facE

# FacenetLocalProject
contains python project to train and test face recognition model. run src/training.py with given parameters and it will train and save classifiers on provided path
