from src.algorithms.runner import AlgorithmRunner


def main() -> None:
	ar = AlgorithmRunner()
	ar.run()


if __name__ == "__main__":
	main()


""" 
usage: python app.py [-h] -i INPUT -o OUTPUT -a {clahe,um,hef} [--rgb] [--shape SHAPE] [--negate] [--threads THREADS] 
              [--filter-type {1,2,3,4}] [--radius RADIUS] [--amount AMOUNT] [--d0 D0] [--window-size WINDOW_SIZE]
              [--clip-limit CLIP_LIMIT] [--n-iter N_ITER] [--log]

options:
  -h, --help            show this help message and exit
  -i, --input INPUT     Path to the images to be processed
  -o, --output OUTPUT   Output path to export the results
  -a, --algorithm {clahe,um,hef}
                        Algorithm to be used
  --rgb                 Convert output images to RGB. Default is False
  --shape SHAPE         Output shape for the processed images, e.g., 640
  --negate              Apply image negation before processing. Default is False
  --threads THREADS     Number of parallel workers for processing images. Default is 1
  --filter-type {1,2,3,4}
                        [UM] Filter type (1=Gaussian, 2=Median, 3=Maximum, 4=Minimum)
  --radius RADIUS       [UM] Radius (sigma) for Gaussian filter
  --amount AMOUNT       [UM] Sharpening amount
  --d0 D0               [HEF] D0 value for high cut (1-90)
  --window-size WINDOW_SIZE
                        [CLAHE] Window size
  --clip-limit CLIP_LIMIT
                        [CLAHE] Clip limit
  --n-iter N_ITER       [CLAHE] Number of iterations
  --log                 [CLAHE] Enable logging intermediate results for debugging. Default is False
"""