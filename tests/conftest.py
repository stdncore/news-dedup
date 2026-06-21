import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: tests that require a downloaded model and are skipped in CI",
    )
