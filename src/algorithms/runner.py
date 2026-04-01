import timeit
from typing import Any
from pathlib import Path
from datetime import timedelta
from multiprocessing import Pool, Value
from multiprocessing.sharedctypes import Synchronized

import src.arguments as ah
from src.algorithms.worker import init_worker, process_image


class AlgorithmRunner:

	__slots__ = ("_rgb", "_negate", "_workers", "_algorithm", "_out_shape",  "_params", "_images_path", "_results_path")

	def __init__(self) -> None:
		arg_handler = ah.ArgumentHandler()
		self._rgb: bool = arg_handler.get_rgb
		self._negate: bool = arg_handler.get_negate
		self._workers: int = arg_handler.get_workers
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
		image_paths = list(self._images_path.glob("*.dcm"))
  
		total: int = len(image_paths)
		counter: Synchronized = Value('i', 0)
		init_args = (self._algorithm, self._params, self._results_path, self._negate, self._rgb, self._out_shape, counter, total)

		if self._workers > 1:
			with Pool(self._workers, initializer=init_worker, initargs=init_args) as pool:
				pool.map(process_image, image_paths)
		else:
			init_worker(*init_args)
			for path in image_paths:
				process_image(path)

		print()
		stop: float = timeit.default_timer()
		time: float = stop - start
		self._export_run_info(time)
		print(f"Processing complete. Output saved to {self._results_path}. Runtime: {timedelta(seconds=time)}")


	def _print_run_info(self) -> None:
		"""Prints the information about the current run, including the selected algorithm, parameters, and output settings."""
		print('\n' + '=' * 10 + " RUN INFO " + '=' * 10)
		print(f"Negate: {self._negate}")
		print(f"Threads: {self._workers}")
		if self._out_shape is not None:
			print(f"Output shape: {self._out_shape}x{self._out_shape}")
		print(f"RGB output: {self._rgb}")
		print(f"Algorithm: {self._algorithm.upper()}")
		print(f"Params:")
		for key, value in self._params.items():
			if value is not None:
				print(f"\t{key}: {value}")
		print('=' * 30 + '\n')


	def _export_run_info(self, runtime: float) -> None:
		"""Exports the runtime and parameters of the current run to a text file in the results directory."""
		with open(self._results_path / "runinfo.txt", 'w') as f:
			f.write(f"Runtime: {timedelta(seconds=runtime)}\n")
			f.write(f"Params:\n")
			for key, value in self._params.items():
				if value is not None:
					f.write(f"\t{key}: {value}\n")