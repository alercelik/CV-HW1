import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogram(I):
	# check whether the image is single band or n-band
	shape = I.shape
	dimension = len(shape)

	if dimension == 2:
		R, C = shape

		hist = np.array([np.count_nonzero(I==i) for i in range(256)], dtype=np.uint32)

		return hist
	elif dimension == 3:
		R, C, B = shape

		hist = np.array([[np.count_nonzero(I[:,:,b]==i) for b in range(B)] for i in range(256)], dtype=np.uint32)
		
		return hist
	else:
		print("a 2-D Image is required")
		raise("non_2D_image")
