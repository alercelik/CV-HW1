import argparse

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


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-i1", "--input1", required=True,
		help="path to first input image")
	ap.add_argument("-i2", "--input2", required=True,
		help="path to second input image")
	args = vars(ap.parse_args())

	I1 = cv2.imread(args["input1"], cv2.IMREAD_COLOR)
	I2 = cv2.imread(args["input2"], cv2.IMREAD_COLOR)

	is_i1_read = I1 is not None
	is_i2_read = I2 is not None

	if not is_i1_read and not is_i2_read:
		raise ("Input1 and Input2 image paths are not correct")
	elif not is_i1_read:
		raise ("Input1 image path is not correct")
	elif not is_i2_read:
		raise ("Input2 image path is not correct")

	I1_rgb = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)
	I2_rgb = cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)

	#print(I1)
	#print(I2)

	print("Shape of first image: {}".format(I1.shape))
	print("Shape of second image: {}".format(I2.shape))

	histI1 = histogram(I1_rgb)
	histI2 = histogram(I2_rgb)

	#print("Histogram of first image: {}".format(histI1.shape))
	#print("Histogram of first image: {}".format(histI1))

	#print("Histogram of second image: {}".format(histI2.shape))
	#print("Histogram of second image: {}".format(histI2))

	result = match_histogram(I1_rgb, histI1, I2_rgb, histI2)
	hist_result = histogram(result)

	result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
	resultant_image = np.hstack((I1, I2, result_bgr))

	cv2.namedWindow("First input, Second input, Result", cv2.WINDOW_FREERATIO)
	cv2.imshow("First input, Second input, Result", resultant_image)

	images = [I1_rgb, I2_rgb, result]
	histograms = [histI1, histI2, hist_result]

	#plot
	fig, axes = plt.subplots(4, 3)
	colors = ["r", "g", "b"]

	for col in range(3):
		axes[0, col].imshow(images[col])

		for b in range(3):
			#print(histograms[col][:, b])
			axes[b+1, col].bar(list(range(256)), histograms[col][:, b], color=colors[b])


	"""
	plt.subplot(4, 1, 1)
	plt.imshow(I1)
	#plt.hist(histI1, bins=256)
	#plt.plot(list(range(256)),histI1)
	for b in range(3):
		plt.subplot(4, 1, b+2)
		plt.bar(list(range(256)), histI1[:,b], color=colors[b])
		#plt.hist(histI1[:,b], bins=256, color=colors[b]) 
	plt.title("histogram")
	"""
	plt.show()

