import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift

from .base import BaseAlgorithm
import src.utils as pu


class HEF(BaseAlgorithm):
	"""High-frequency Emphasis filtering algorithm"""

	__slots__ = ("_d0v")

	def __init__(self, d0v: int | None) -> None:
		if d0v is None or not(1 <= d0v <= 90):
			raise ValueError("D0 value must be between 1 and 90!")
		self._d0v: int = d0v


	def run(self, image: np.ndarray) -> np.ndarray:
		if len(image.shape) > 2:
			image = pu.to_grayscale(image)

		image = pu.normalize(np.min(image), np.max(image), 0, 255, image)

		# HF part
		img_fft: np.ndarray = fft2(image)  # img after fourier transformation
		img_sfft: np.ndarray = fftshift(img_fft)  # img after shifting component to the center

		m, n = img_sfft.shape
		filter_array: np.ndarray = np.zeros((m, n))

		for i in range(m):
			for j in range(n):
				filter_array[i, j] = 1.0 - np.exp(- ((i-m / 2.0) ** 2 + (j-n / 2.0) ** 2) / (2 * (self._d0v ** 2)))
		k1: float = 0.5
		k2: float = 0.75
		high_filter = k1 + k2 * filter_array

		img_filtered = high_filter * img_sfft
		img_hef: np.ndarray = np.real(ifft2(fftshift(img_filtered)))  # HFE filtering done

		# Normalize float result back to [0, 255] before histogram equalization
		img_hef_normalized: np.ndarray = pu.normalize(np.min(img_hef), np.max(img_hef), 0, 255, img_hef)

		# HE part
		hist, bins = pu.histogram(img_hef_normalized)
		hist_eq: dict[int, int] = pu.calculate_cdf(hist, bins)

		result: np.ndarray = np.zeros((m, n), dtype=np.uint8)
		for i in range(m):
			for j in range(n):
				result[i][j] = hist_eq[img_hef_normalized[i][j]]

		return result.astype(np.uint8)