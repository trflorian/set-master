from pathlib import Path

import pytest


@pytest.fixture
def output_dir() -> Path:
    """Provide the shared, gitignored directory for generated test artifacts."""
    path = Path(__file__).parent / "output"
    path.mkdir(parents=True, exist_ok=True)
    return path
