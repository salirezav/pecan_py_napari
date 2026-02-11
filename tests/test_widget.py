"""Minimal test for the napari-pecan-py widget."""

import pytest

from napari_pecan_py._widget import hello_widget


def test_hello_widget_import():
    """Widget function is importable and callable."""
    assert callable(hello_widget)
    hello_widget()
