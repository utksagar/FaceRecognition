import tensorflow as tf
import detect_face
from scipy import misc
import numpy as np

class AlignMTCNN:
    PER_PROCESS_GPU_MEMORY_FRACTION = 0.2
    MINSIZE = 20 # minimum size of face
    THRESHOLD = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    FACTOR = 0.709 # scale factor
    MARGIN = 40
    def __init__(self):
        return None
    def get_session(self, gpu_memory = PER_PROCESS_GPU_MEMORY_FRACTION):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = gpu_memory)
            sess = tf.Session(config=tf.ConfigProto(gpu_options = gpu_options, log_device_placement = False))
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
        return pnet, rnet, onet
    
    def detect_face(self, img,  pnet, rnet, onet,largetsbb = True, minsize = MINSIZE, threshold = THRESHOLD, factor = FACTOR):
        #img = misc.imread(img_path)
        if img.ndim == 2:
            img = to_rgb(img)
        img = img[:,:,0:3]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if largetsbb:
            return self.getLargestFaceBoundingBox(bounding_boxes)
        else:
            return bounding_boxes[:,:-1]
        
    def to_rgb(self, img):
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret

    def getLargestFaceBoundingBox(self, bb):
        assert bb is not None

        if len(bb) != 1:
            return np.expand_dims(max(np.squeeze(bb[:,:-1]), key=lambda x: x[2]-x[0] * x[3]-x[1]), axis=0)
        else:
            return bb[:,:-1]
        
    def aligned(self, imgsize, img, pnet, rnet, onet, largetsbb = True, minsize = MINSIZE, threshold = THRESHOLD, factor = FACTOR):
        bbox = self.detect_face(img, pnet, rnet, onet,largetsbb = largetsbb, minsize = minsize, threshold = threshold, factor = factor)
        margin = self.MARGIN
        scaled_img = []
        img_size = np.asarray(img.shape)[0:2]
        for bb1 in bbox:
            det = np.squeeze(bb1)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            scaled = misc.imresize(cropped, (imgsize, imgsize), interp='bilinear')
            scaled_img.append(scaled)
        return scaled_img