import imageio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import src.utils as pu
from src.algorithms.base import BaseAlgorithm


class CLAHE(BaseAlgorithm):
	"""Contrast Limited Adaptive Histogram Equalization.
	In reality, we do a normalization before applying CLAHE, making it the N-CLAHE method, but in
	N-CLAHE the normalization is done using a log function, instead of a linear one, as we use here.
	"""

	__slots__ = ("_n_iter", "_clip_limit", "_window_size", "_result_path", "_log")

	def __init__(self, result_path: Path, window_size: int | None, clip_limit: int | None, n_iter: int | None, log: bool = False) -> None:
		if window_size is None or clip_limit is None or n_iter is None:
			raise ValueError("CLAHE parameters must be provided!")

		self._log: bool = log
		self._n_iter: int = n_iter
		self._clip_limit: int = clip_limit
		self._window_size: int = window_size
		self._result_path: Path = result_path / "clahe_log"
		self._result_path.mkdir(parents=True, exist_ok=True)


	def run(self, image: np.ndarray) -> np.ndarray:
		if len(image.shape) > 2:
			image = pu.to_grayscale(image)

		normalized_image: np.ndarray = pu.normalize(np.min(image), np.max(image), 0, 255, image)
		if self._log:
			imageio.imwrite(self._result_path / "normalized_image.png", normalized_image)

		equalized_image: np.ndarray = self._clahe(normalized_image)
		if self._log:
			self._export_histogram(image, normalized_image, equalized_image)

		return equalized_image


	def _clahe(self, image: np.ndarray) -> np.ndarray:
		"""Applies the CLAHE algorithm to the given image.

		Args:
			image (np.ndarray): Input image to be processed.

		Returns:
			np.ndarray: CLAHE result.
		"""
		border: int = self._window_size // 2
		padded_image: np.ndarray = np.pad(image, border, "reflect")

		shape: tuple[int, int] = padded_image.shape
		padded_equalized_image: np.ndarray = np.zeros(shape).astype(np.uint8)

		for i in range(border, shape[0] - border):
			print(f"Line: {i}", end='\r', flush=True)
			for j in range(border, shape[1] - border):
				# Region to extract the histogram
				region: np.ndarray = padded_image[i-border:i+border+1, j-border:j+border+1]

				# Calculating the histogram from region
				hist, bins = pu.histogram(region)

				# Clipping the histogram
				pu.clip_histogram(hist, bins, self._clip_limit)

				# Trying to reduce the values above clipping
				for _ in range(self._n_iter):
					pu.clip_histogram(hist, bins, self._clip_limit)

				# Calculating the CDF
				cdf = pu.calculate_cdf(hist, bins)

				# Changing the value of the image to the result from the CDF for the given pixel
				padded_equalized_image[i][j] = cdf[padded_image[i][j]]

		print()
		# Removing the padding from the image
		return padded_equalized_image[border:shape[0] - border, border:shape[1] - border].astype(np.uint8)


	def _export_histogram(self, image: np.ndarray, normalized: np.ndarray, equalized: np.ndarray) -> None:
		"""Exports the histograms of the original, normalized, and equalized images to a PNG file in the results directory."""
		plt.xlabel("Pixel")
		plt.ylabel("Count")

		hist, bins = pu.histogram(image)
		plt.plot(bins, hist, label='Original Image')
		plt.legend()

		hist, bins = pu.histogram(normalized)
		plt.plot(bins, hist, label='Normalized Image')
		plt.legend()

		hist, bins = pu.histogram(equalized)
		plt.plot(bins, hist, label='CLAHE Result')
		plt.legend()
		plt.savefig(self._result_path / "histograms.png")