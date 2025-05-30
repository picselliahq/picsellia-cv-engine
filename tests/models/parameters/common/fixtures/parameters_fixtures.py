import pytest

from picsellia_cv_engine.models.parameters.common.parameters import Parameters


class ConcreteParameters(Parameters):
    """A concrete implementation of the abstract Parameters class for testing."""

    pass


@pytest.fixture
def log_data():
    return {
        "key1": "value1",
        "key2": 100,
        "key3": True,
        "key4": {"nested_key1": 63, "nested_key2": 42.0},
        "key5": "null",
        "key6": None,
    }


@pytest.fixture
def parameters(log_data):
    return ConcreteParameters(log_data)
