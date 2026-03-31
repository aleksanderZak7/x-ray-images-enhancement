import numpy as np
from skimage import img_as_float
from scipy.ndimage import gaussian_filter, median_filter, maximum_filter, minimum_filter

from .base import BaseAlgorithm


class UM(BaseAlgorithm):

	__slots__ = ("_amount", "_filter", "_radius")

	def __init__(self, filter_type: int | None, amount: int | None, radius: int | None) -> None:
		if filter_type is None or amount is None:
			raise ValueError("UM parameters must be provided!")
		if filter_type == 1 and radius is None:
			raise ValueError("Radius parameter must be provided for Gaussian filter!")

		self._amount: int = amount
		self._filter: int = filter_type
		self._radius: int | None = radius


	def run(self, image: np.ndarray) -> np.ndarray:
		image = img_as_float(image) # ensuring float values for computations

		if self._filter == 1:
			blurred_image = gaussian_filter(image, sigma=self._radius)

		elif self._filter == 2:
			blurred_image = median_filter(image, size=20)

		elif self._filter == 3:
			blurred_image = maximum_filter(image, size=20)

		else:
			blurred_image = minimum_filter(image, size=20)

		# keep the edges created by the filter
		mask = image - blurred_image
		sharpened_image = image + mask * self._amount

		# Interval [0.0, 1.0]
		sharpened_image = np.clip(sharpened_image, float(0), float(1))

		# Interval [0,255]
		return (sharpened_image * 255).astype(np.uint8)