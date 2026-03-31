import json
import timeit
import imageio
import pydicom
import numpy as np
from typing import Any
from pathlib import Path
from datetime import timedelta
from skimage.transform import resize

import src.arguments as ah
from src.algorithms.hef import HEF
from src.algorithms.clahe import CLAHE
from src.algorithms.unsharping_mask import UM


class AlgorithmRunner:

	__slots__ = ("_negate", "_algorithm", "_out_shape", "_images_path", "_results_path", "_params")

	def __init__(self) -> None:
		arg_handler = ah.ArgumentHandler()
		self._negate: bool = arg_handler.get_negate
		self._algorithm: str = arg_handler.get_algorithm
		self._out_shape: int | None = arg_handler.get_shape
		self._params: dict[str, Any] = arg_handler.get_params
  
		self._images_path: Path = arg_handler.get_input_path
		self._results_path: Path = arg_handler.get_output_path

		self._results_path.mkdir(parents=True, exist_ok=True)


	def run(self) -> None:
		"""Runs the selected algorithm for the *.dcm files in provided path."""
		self._print_run_info()
		start: float = timeit.default_timer()
		alg: UM | HEF | CLAHE = self._select_algorithm()

		for image_path in self._images_path.glob("*.dcm"):
			dataset: pydicom.FileDataset = pydicom.dcmread(image_path)

			pixel_matrix: np.ndarray = dataset.pixel_array
			if self._negate:
				pixel_matrix = np.max(pixel_matrix) - pixel_matrix

			processed_image = alg.run(pixel_matrix)

			if self._out_shape is not None:
				processed_image = (resize(processed_image, (self._out_shape, self._out_shape)) * 255).astype(np.uint8) # type: ignore
			imageio.imwrite(self._results_path / f"{image_path.stem}.png", processed_image)

		stop: float = timeit.default_timer()
		time: float = stop - start
		self._export_run_info(time)
		print(f"Processing complete. Output saved to {self._results_path}. Runtime: {timedelta(seconds=time)}")


	def _print_run_info(self) -> None:
		"""Prints the information about the current run, including the selected algorithm, parameters, and output settings."""
		print('\n' + '=' * 10 + " RUN INFO " + '=' * 10)
		print(f"Algorithm: {self._algorithm.upper()}")
		print(f"Params:")
		for key, value in self._params.items():
			if value is not None:
				print(f"\t{key}: {value}")

		print(f"Negate: {self._negate}")
		if self._out_shape is not None:
			print(f"Output shape: {self._out_shape}x{self._out_shape}")
		print('=' * 30 + '\n')


	def _select_algorithm(self) -> UM | HEF | CLAHE:
		"""Selects the algorithm to be used based on the provided argument.

		Raises:
			ValueError: If the provided algorithm is not supported.

		Returns:
			UM | HEF | CLAHE: An instance of the selected algorithm class.
		"""
		if self._algorithm == 'um':
			return UM(self._params["filter_type"], self._params["amount"], self._params["radius"])

		elif self._algorithm == 'hef':
			return HEF(self._params["d0"])

		elif self._algorithm == 'clahe':
			return CLAHE(self._results_path, self._params["window_size"], 
                self._params["clip_limit"], self._params["n_iter"], self._params["log"])

		else:
			raise ValueError(f"Algorithm {self._algorithm} is not supported!")


	def _export_run_info(self, runtime: float) -> None:
		"""Exports the runtime and parameters of the current run to a text file in the results directory."""
		with open(self._results_path / "runinfo.txt", 'w') as f:
			f.write(f"Runtime: {timedelta(seconds=runtime)}\n")
			f.write(f"Params:\n{json.dumps(self._params)}")