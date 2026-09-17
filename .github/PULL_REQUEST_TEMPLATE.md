## Summary

<!-- What does this change do, and why? -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Performance
- [ ] Refactor / internal cleanup
- [ ] Build, CI or tooling

## Testing

<!-- Which commands did you run, and what was the result? -->

```
./scripts/test.sh
./scripts/lint.sh
```

## Checklist

- [ ] New behaviour is covered by tests, and the full suite passes on CPU.
- [ ] `ruff check .` and `ruff format --check .` pass.
- [ ] Public functions and classes have type hints and docstrings.
- [ ] Nothing in the change requires a GPU, CUDA, TensorFlow or an API key to
      run the core test suite.
- [ ] No credentials, tokens or private data are included.
- [ ] Documentation and `CHANGELOG.md` are updated where relevant.
- [ ] Anything not yet implemented is described as planned, not as available.

## CUDA changes only

- [ ] Kernel results were compared against the NumPy backend.
- [ ] Kernels use bounds checking and CUDA error checking.
- [ ] The CPU path still works with the extension absent.
