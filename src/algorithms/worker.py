import imageio
import pydicom
import numpy as np
from typing import Any
from pathlib import Path
from skimage.transform import resize
from multiprocessing.sharedctypes import Synchronized

from src.algorithms.hef import HEF
from src.algorithms.clahe import CLAHE
from src.algorithms.unsharping_mask import UM


_total: int = 0
_rgb: bool = False
_negate: bool = False
_out_shape: int | None = None
_results_path: Path | None = None
_counter: Synchronized | None = None
_alg: UM | HEF | CLAHE | None = None


def init_worker(algorithm: str, params: dict[str, Any], results_path: Path, negate: bool, rgb: bool,
				out_shape: int | None, counter: Synchronized, total: int) -> None:
	"""Initializer for each worker process — creates one algorithm instance per worker."""
	global _alg, _rgb, _negate, _out_shape, _results_path, _counter, _total
	_rgb = rgb
	_total = total
	_negate = negate
	_counter = counter
	_out_shape = out_shape
	_results_path = results_path

	if algorithm == 'um':
		_alg = UM(params["filter_type"], params["amount"], params["radius"])
	elif algorithm == 'hef':
		_alg = HEF(params["d0"])
	elif algorithm == 'clahe':
		_alg = CLAHE(results_path, params["window_size"], params["clip_limit"], params["n_iter"], params["log"])
	else:
		raise ValueError(f"Algorithm {algorithm} is not supported!")


def process_image(image_path: Path) -> None:
	"""Processes a single image using the worker's algorithm instance."""
	assert _alg is not None and _results_path is not None and _counter is not None, "Worker not properly initialized!"
	dataset: pydicom.FileDataset = pydicom.dcmread(image_path)
	pixel_matrix: np.ndarray = dataset.pixel_array

	if _negate:
		pixel_matrix = np.max(pixel_matrix) - pixel_matrix

	processed_image = _alg.run(pixel_matrix)

	if _out_shape is not None:
		processed_image = (resize(processed_image, (_out_shape, _out_shape)) * 255).astype(np.uint8)  # type: ignore
	if _rgb:
		processed_image = np.stack([processed_image] * 3, axis=-1)
	imageio.imwrite(_results_path / f"{image_path.stem}.png", processed_image)

	with _counter.get_lock():
		_counter.value += 1
		print(f"\r[{_counter.value}/{_total}] Processed: {image_path.name}\033[K", end='', flush=True)