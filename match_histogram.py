import cv2
import numpy as np
import matplotlib.pyplot as plt


def prepare_lut_for_histogram_matching(cdf1, cdf2, B):
	lut = np.zeros((256, B), np.uint8)

	for b in range(B):
		j = 0

		#lut = np.array( [j for j in 256] )

		for i in range(256):
			while cdf2[j, b] < cdf1[i, b] and j < 255:
				j += 1
			lut[i, b] = j

		#print(lut[:, b])
	return lut


def match_histogram(source, hist_source, target, hist_target):
	shape_source = source.shape
	dimension_source = len(shape_source)

	shape_target = target.shape
	dimension_target = len(shape_target)

	if not dimension_source == dimension_target:
		raise("dimesion of source and target do not match")

	if dimension_source == 2:
		B = 2
	if dimension_source == 3:
		B = 3
		R, C, B = shape_source

		pdf_source = hist_source / (shape_source[0] * shape_source[1])
		cdf_source = pdf_source.cumsum(axis=0)

		pdf_target = hist_target / (shape_target[0] * shape_target[1])
		cdf_target = pdf_target.cumsum(axis=0)

		lut = prepare_lut_for_histogram_matching(cdf_source, cdf_target, B)

		#print(lut.shape, lut.dtype)

		#resultant_image = lut[source].astype("uint8")  # resulted in 4d image
		#resultant_image = cv2.LUT(source, lut)  # did not work
		#resultant_image = np.array( [[[ lut[ source[r, c, b], b ] for b in range(B)] for c in range(C)] for r in range(R)] , dtype=np.uint8)  # slow for large images
		#resultant_image = lut[source[:,:], ...]  # close but did not work, resulted in 4d image
		#resultant_image = lut[source[:,:, ...], ...]  # close but did not work, resulted in 4d image

		
		resultant_image = np.zeros(source.shape, np.uint8)
		for b in range(B):
			resultant_image[:,:,b] = lut[source[:,:,b], b]

		"""
		print(resultant_image.shape)
		print(resultant_image)

		cv2.imshow("original", source)
		cv2.imshow("target", target)

		cv2.imshow("resultant_image", resultant_image)
		cv2.waitKey()
		"""

		return resultant_image
