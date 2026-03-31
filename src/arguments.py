import argparse
from typing import Any
from pathlib import Path


class ArgumentHandler:
	"""Handles the program's arguments."""

	__slots__ = ("_parser", "_parsed_args")

	def __init__(self) -> None:
		self._parser = argparse.ArgumentParser()

		self._define_arguments()
		self._parsed_args = vars(self._parser.parse_args())


	def _define_arguments(self) -> None:
		"""Defines the command-line arguments for the program."""
		self._parser.add_argument("-i", "--input", type=Path, required=True, help="Path to the images to be processed")
		self._parser.add_argument("-o", "--output", type=Path, required=True, help="Output path to export the results")
		self._parser.add_argument("-a", "--algorithm", type=str, required=True, choices=["clahe", "um", "hef"], help="Algorithm to be used")

		# Image processing options
		self._parser.add_argument("--negate", action="store_true", help="Apply image negation before processing")
		self._parser.add_argument("--shape", type=int, default=None, help="Output shape for the processed images, e.g., 640")

		# UM parameters
		self._parser.add_argument("--filter-type", type=int, choices=[1, 2, 3, 4], default=None, help="[UM] Filter type (1=Gaussian, 2=Median, 3=Maximum, 4=Minimum)")
		self._parser.add_argument("--radius", type=int, default=None, help="[UM] Radius (sigma) for Gaussian filter")
		self._parser.add_argument("--amount", type=int, default=None, help="[UM] Sharpening amount")

		# HEF parameters
		self._parser.add_argument("--d0", type=int, default=None, help="[HEF] D0 value for high cut (1-90)")

		# CLAHE parameters
		self._parser.add_argument("--window-size", type=int, default=None, help="[CLAHE] Window size")
		self._parser.add_argument("--clip-limit", type=int, default=None, help="[CLAHE] Clip limit")
		self._parser.add_argument("--n-iter", type=int, default=None, help="[CLAHE] Number of iterations")
		self._parser.add_argument("--log", action="store_true", help="[CLAHE] Enable logging intermediate results for debugging")


	@property
	def get_input_path(self) -> Path:
		"""Returns provided input path."""
		return self._parsed_args["input"]

	@property
	def get_output_path(self) -> Path:
		"""Returns provided output path."""
		return self._parsed_args["output"]

	@property
	def get_algorithm(self) -> str:
		"""Returns provided algorithm."""
		return self._parsed_args["algorithm"]

	@property
	def get_negate(self) -> bool:
		"""Returns whether image negation should be applied."""
		return self._parsed_args["negate"]

	@property
	def get_shape(self) -> int | None:
		"""Returns provided output shape for the processed images."""
		return self._parsed_args["shape"]

	@property
	def get_params(self) -> dict[str, Any]:
		"""Returns algorithm-specific parameters."""
		return {
			"filter_type": self._parsed_args["filter_type"],
			"radius": self._parsed_args["radius"],
			"amount": self._parsed_args["amount"],
			"d0": self._parsed_args["d0"],
			"window_size": self._parsed_args["window_size"],
			"clip_limit": self._parsed_args["clip_limit"],
			"n_iter": self._parsed_args["n_iter"],
			"log": self._parsed_args["log"]
		}