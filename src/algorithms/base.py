import numpy as np
from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
	"""Class that must be inherited for each algorithm."""

	@abstractmethod
	def run(self, image: np.ndarray) -> np.ndarray:
		"""Runs the algorithm for the image.

		Args:
			image (np.ndarray): Input image to be processed.

		Raises:
			NotImplementedError: If the method is not implemented in the child class.

		Returns:
			np.ndarray: Processed image after applying the algorithm.
		"""
		raise NotImplementedError