# Benchmarks

Scripts that measure EveryO against the other backends available on the
machine. Every number is produced by an actual timed run — nothing in this
directory is hardcoded, and results are never committed.

```bash
python benchmarks/benchmark_matmul.py --sizes 128 256 512
python benchmarks/benchmark_relu.py
python benchmarks/benchmark_training.py --epochs 10
```

Or through the CLI:

```bash
everyo benchmark --sizes 128 256 512 --output results.json
```

## Method

* Warm-up runs are executed and discarded before timing, so lazy imports,
  memory allocation and (on a GPU) kernel compilation do not pollute the
  measurement.
* Each measurement is repeated; the mean, best and worst times are all
  recorded.
* TensorFlow results call `.numpy()` inside the timed region, which forces the
  asynchronous computation to complete.
* CUDA timings include host-to-device and device-to-host transfers, because
  that is what an EveryO user actually pays. At small sizes transfer cost can
  exceed the compute saving — the point of benchmarking is to find out where
  that crossover is on your hardware rather than assuming the GPU wins.

## Exporting

`--output results.json` writes the rows together with a description of the
machine (platform, CPU, library versions, whether a GPU was detected).
`--output results.csv` writes the rows alone. `--plot chart.png` renders a
log-log chart of time against problem size.
