# Contributing to EveryO

Thanks for considering a contribution. EveryO is a small project; issues and
pull requests are welcome, and there is no bureaucracy to work through.

## Development setup

```bash
git clone https://github.com/krishanth7/EveryO.git
cd EveryO

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

pip install -e ".[dev]"
```

Or run `./scripts/setup.sh`, which does the same and then prints
`everyo info`.

Optional extras:

```bash
pip install -e ".[tensorflow]"   # the TensorFlow backend (a large download)
pip install -e ".[cuda]"         # pybind11, for building the CUDA extension
```

## Branch workflow

1. Fork the repository and create a branch from `main`:
   `git checkout -b fix/dropout-scaling`.
2. Make your change with tests.
3. Run the checks below.
4. Open a pull request describing what changed and why.

Branch names like `fix/...`, `feature/...` or `docs/...` are appreciated but
not enforced.

## Running the tests

```bash
./scripts/test.sh                # everything
./scripts/test.sh --fast         # skip tests marked 'slow'
./scripts/test.sh --coverage     # with a coverage report
./scripts/test.sh tests/test_autograd.py -k gradient
```

The full suite must pass on CPU with neither TensorFlow nor CUDA installed.
Tests for optional components use `pytest.mark.skipif` so they skip cleanly.

## Linting and formatting

```bash
./scripts/lint.sh                # report problems
./scripts/lint.sh --fix          # fix and format
```

CI runs `ruff check .` and `ruff format --check .`, so run `--fix` before
pushing.

## Coding conventions

* Type hints on public functions and methods.
* A docstring on every public module, class and function, saying what it does
  and what it raises when that is not obvious.
* Small modules with one clear responsibility. If a file is growing past a few
  hundred lines, it probably wants splitting.
* Errors should name the values involved:
  `EveryOShapeError: Cannot multiply matrices with shapes (32, 64) and (128, 10).`
  rather than a bare `IndexError`.
* Keep the `everyo.core` layer dependency-free beyond NumPy.
* No feature may require an API key, a network call, a GPU or TensorFlow in
  order for the core to work.

## Adding an operation

1. Implement the forward pass in `everyo/core/operations.py` and return the
   input gradients from the `backward_fn` closure.
2. Add a finite-difference check to `tests/test_autograd.py` — this is not
   optional, it is how gradient bugs are caught.
3. Export it from `everyo/__init__.py`.

## Adding a layer

1. Subclass `Module`, decorate with `@register_module`, implement `forward`.
2. Implement `get_config()` when the constructor takes arguments, so the layer
   can be serialised.
3. Export it from `everyo/nn/__init__.py` and `everyo/__init__.py`.
4. Add tests, including a save/load round trip.

## CUDA development

CUDA changes need extra care because most contributors and all hosted CI
runners have no GPU.

* Write the kernel in `cuda/src/`, declare it in `cuda/include/everyo_cuda.h`
  and expose it in `cuda/bindings/bindings.cpp`.
* Use `EVERYO_CUDA_CHECK` around every CUDA call and
  `EVERYO_CUDA_CHECK_KERNEL()` after every launch.
* Bounds-check inside the kernel: grids are rounded up, so the last block
  overshoots.
* Own device memory with `detail::DeviceBuffer` so it is freed on exceptions.
* Add a NumPy fallback in `everyo/cuda/interface.py`.
* Add a correctness test to `tests/test_devices.py::TestCudaKernels` that
  compares against NumPy and skips without a GPU.
* Build and test locally: `./scripts/build_cuda.sh && pytest tests/test_devices.py -v`.
  Say clearly in the pull request which hardware you tested on.

## Documentation

* Update `docs/` when behaviour changes.
* Update `README.md` only for user-visible changes.
* Never describe planned functionality as available. The README separates
  Available, Experimental and Planned, and that separation should stay honest.
* Update `CHANGELOG.md` under "Unreleased".

## Reporting issues

Include the output of `everyo info`, a minimal reproduction and the full
traceback. Please do not paste credentials or private data — EveryO never needs
any.

## Scope

EveryO aims to be a *readable* framework that still works properly. A change
that doubles performance while making a module unreadable is probably the wrong
trade for this project; a change that makes something clearer without breaking
correctness is almost always welcome.
