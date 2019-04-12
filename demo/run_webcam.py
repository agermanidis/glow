from __future__ import print_function
from __future__ import division

import sys
sys.path.insert(0, 'src')
import argparse
import numpy as np
#import transform, vgg, pdb, os
import tensorflow as tf
import cv2
from datetime import datetime

from model import *
from imutils.face_utils.helpers import FACIAL_LANDMARKS_68_IDXS
from imutils.face_utils.helpers import FACIAL_LANDMARKS_5_IDXS
from imutils.face_utils.helpers import shape_to_np
import numpy as np
import cv2
import dlib



# parser
parser = argparse.ArgumentParser()
parser.add_argument('--device_id', type=int, help='camera device id (default 0)', required=False, default=0)
parser.add_argument('--width', type=int, help='width to resize camera feed to (default 320)', required=False, default=640)
parser.add_argument('--disp_width', type=int, help='width to display output (default 640)', required=False, default=1200)
parser.add_argument('--disp_source', type=int, help='whether to display content and style images next to output, default 1', required=False, default=1)
parser.add_argument('--horizontal', type=int, help='whether to concatenate horizontally (1) or vertically(0)', required=False, default=1)
parser.add_argument('--num_sec', type=int, help='number of seconds to hold current model before going to next (-1 to disable)', required=False, default=-1)



class FaceWarper:
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=512, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, gray, rect, z_addition):
        # convert the landmark (x, y)-coordinates to a NumPy
        h1, w1 = image.shape[:2]
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)

        #simple hack ;)
        if (len(shape)==68):
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        else:
            (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]

        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC)

        # invert the previous affine transformation for later
        Mi = cv2.invertAffineTransform(M)

        # BGR -> RGB
        output = output[:,:,::-1]

        # encode with GLOW, do operations on z
        z = encode(output)
        z[0] += z_addition

        # decode back to image and back to BGR
        output = decode(z)[0]
        output = output[:,:,::-1]

        # invert the affine transformation on output
        output = cv2.warpAffine(output, Mi, (w1, h1),
            flags=cv2.INTER_CUBIC)

        # overwrite original image with masked output
        mask = np.sum(output, axis=2) == 0.0
        image = np.multiply(mask.reshape((h1, w1, 1)), image)
        image += output

        return image


def get_camera_shape(cam):
	""" use a different syntax to get video size in OpenCV 2 and OpenCV 3 """
	cv_version_major, _, _ = cv2.__version__.split('.')
	if cv_version_major == '3' or cv_version_major == '4':
		return cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
	else:
		return cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH), cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)


def main(device_id, width, disp_width, disp_source, horizontal, num_sec):

	# load face detection and warping
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
	warper = FaceWarper(predictor, desiredFaceWidth=256, desiredLeftEye=(0.371, 0.480))

	# tags that can be modified
	tags = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
	tags = tags.split()
	selected_tags = "Young Attractive Blond_Hair Eyeglasses Goatee Gray_Hair Heavy_Makeup Male Bald Bangs Big_Lips Big_Nose 5_o_Clock_Shadow High_Cheekbones Arched_Eyebrows Bags_Under_Eyes Black_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie ".split()
	idx_t = 0
	z_mult = 1
	z_mag = 0.75

	# load camera and cv
	cam = cv2.VideoCapture(device_id)
	cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
	cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, 1)
	cam_width, cam_height = get_camera_shape(cam)

	# enter cam loop
	while True:
		ret, frame = cam.read()
		#frame = cv2.resize(frame, (width, height))
		frame = cv2.flip(frame, 1)
		img = np.copy(frame)

		z_addition = z_mult * z_mag * z_manipulate[tags.index(selected_tags[idx_t])]
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		rects = detector(gray, 2)
		if len(rects) > 0:
			img = warper.align(img[:, :, ::-1], gray, rects[0], z_addition)[:, :, ::-1]
			img = np.array(Image.fromarray(img).convert('RGB'))
			img = np.clip(img, 0, 255).astype(np.uint8)

		output = np.concatenate([frame, img], axis=1)
		cv2.imshow('frame', output)
		#output = cv2.resize(output, (disp_width, int(oh * disp_width / ow)))
		
		key_ = cv2.waitKey(1)	
		if key_ == 27:
			break
		elif key_ == ord('a'):
			idx_t = (idx_t + len(selected_tags) - 1) % len(selected_tags)
		elif key_ == ord('s'):
			idx_t = (idx_t + 1) % len(selected_tags)
		elif key_ == ord('z'):
			z_mag = max(0, z_mag-0.05)
		elif key_ == ord('x'):
			z_mag = min(3, z_mag+0.05)
		elif key_ == ord('c'):
			z_mult *= -1

	# done
	cam.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	opts = parser.parse_args()
	main(opts.device_id, opts.width, opts.disp_width, opts.disp_source==1, opts.horizontal==1, opts.num_sec),

