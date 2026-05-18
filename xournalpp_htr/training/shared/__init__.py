"""Shared training/inference utilities (ADR 006 section 3).

This package contains utilities shared across models. Per ADR 006 it has *no*
dependencies beyond the base package, so it is importable in the lean
inference-only install (it is deliberately *not* guarded like the per-model
training subpackages).
"""
