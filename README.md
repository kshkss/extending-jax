# Extending JAX with custom Rust code
This repository demonstrate how to add custom XLA ops to JAX with custom Rust code.
The codes are based on [demonstration using c++ and CUDA](https://github.com/dfm/extending-jax).

# Running the demonstration
This repository uses [Rye](https://rye-up.com/) to manage python environment
and [Pytest](https://docs.pytest.org/en/7.4.x/contents.html) to manage tests.
First of all, compile Rust codes and set up python environment:
```
# rye sync
```
Then, run tests:
```
# rye run pytest
```

