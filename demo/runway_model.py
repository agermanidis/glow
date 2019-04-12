import runway
from model import *
from imutils.face_utils.helpers import FACIAL_LANDMARKS_68_IDXS
from imutils.face_utils.helpers import FACIAL_LANDMARKS_5_IDXS
from imutils.face_utils.helpers import shape_to_np
import numpy as np
import cv2
import dlib


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



# load face detection and warping
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
warper = FaceWarper(predictor, desiredFaceWidth=256, desiredLeftEye=(0.371, 0.480))

# tags that can be modified
tags = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
tags = tags.split()

# launch Runway
model_ready = False

@runway.setup(options={"model_name": runway.text })
def setup():
	global model_ready
	print('setup model')
	model_ready = True
	return None


@runway.command('convert', inputs={'image': runway.image, 'feature': runway.category(choices=tags, default=tags[2]), 'amount': runway.number}, outputs={'output': runway.image})
def detect(sess, inp):
	img = np.array(inp['image'])
    amount = inp['amount']
    feature = inp['feature']
	z_addition = amount * z_manipulate[tags.index(feature)]
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 2)
	if len(rects) == 0 or not model_ready:
		print('nothing found')
		return dict(output=img)
	img = warper.align(img[:, :, ::-1], gray, rects[0], z_addition)[:, :, ::-1]
	img = np.array(Image.fromarray(img).convert('RGB'))
	output = np.clip(img, 0, 255).astype(np.uint8)
	return dict(output=output)


if __name__ == '__main__':
    runway.run()
