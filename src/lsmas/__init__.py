"""Parameter reduction for calibrating the SIPNET land-surface model at Harvard Forest."""

# EnsKit turns on JAX's 64-bit mode when it is imported. Importing it before anything builds a JAX
# array keeps every array in the package at one precision.
import enskit  # noqa: F401  # isort: skip
