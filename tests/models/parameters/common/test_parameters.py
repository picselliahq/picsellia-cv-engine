from enum import Enum
from unittest.mock import patch

import pytest

from tests.models.parameters.common.fixtures.parameters_fixtures import (
    ConcreteParameters,
)


class TestTileMode(Enum):
    CONSTANT = 0
    DROP = 1
    REFLECT = 2
    EDGE = 3
    WRAP = 4


class TestParameters:
    def test_extract_parameter(self, parameters):
        assert (
            parameters.extract_parameter(keys=["key1"], expected_type=str) == "value1"
        )
        assert parameters.extract_parameter(keys=["key2"], expected_type=int) == 100
        assert parameters.extract_parameter(keys=["key3"], expected_type=bool) is True
        assert parameters.extract_parameter(keys=["key4"], expected_type=dict) == {
            "nested_key1": 63,
            "nested_key2": 42.0,
        }

    def test_extract_parameter_with_default(self, parameters):
        assert (
            parameters.extract_parameter(
                keys=["nonexistent"], expected_type=int, default=42
            )
            == 42
        )

        assert (
            parameters.extract_parameter(
                keys=["nonexistent"], expected_type=str, default="value"
            )
            == "value"
        )

        assert parameters.extract_parameter(
            keys=["nonexistent"], expected_type=bool, default=True
        )

    def test_extract_parameter_with_default_value_and_wrong_expected_type_raises_an_error(
        self, parameters
    ):
        with pytest.raises(TypeError):
            parameters.extract_parameter(
                keys=["nonexistent"], expected_type=int, default="42"
            )

    def test_extract_parameter_outside_range_raises_an_error(self, parameters):
        with pytest.raises(ValueError):
            parameters.extract_parameter(
                keys=["key2"], expected_type=int, range_value=(101, 200)
            )

    def test_extract_parameter_int_to_float_conversion(self, parameters):
        assert parameters.extract_parameter(keys=["key2"], expected_type=float) == 100.0

    def test_extract_parameter_missing_key_without_default_raises_an_error(
        self, parameters
    ):
        with pytest.raises(KeyError):
            parameters.extract_parameter(keys=["nonexistent"], expected_type=str)

    def test_extract_parameter_missing_key_with_range_without_default_raises_an_error(
        self, parameters
    ):
        with pytest.raises(KeyError):
            parameters.extract_parameter(
                keys=["nonexistent"], expected_type=int, range_value=(0, 100)
            )

    def test_extract_parameter_without_provided_keys_raises_an_error(self, parameters):
        with pytest.raises(ValueError):
            parameters.extract_parameter(keys=[], expected_type=str)

    def test_extract_parameter_when_type_check_returns_none_raises_an_error(
        self, parameters
    ):
        with patch.object(parameters, "_flexible_type_check", return_value=None):
            with pytest.raises(TypeError):
                parameters.extract_parameter(keys=["key1"], expected_type=str)

    def test_to_dict_excludes_certain_attributes(self, parameters):
        param_dict = parameters.to_dict()
        assert "parameters_data" not in param_dict
        assert "defaulted_keys" not in param_dict

    def test_create_parameters_with_invalid_context_raises_an_error(self):
        with pytest.raises(ValueError):
            ConcreteParameters(log_data="not_a_dict")

    @pytest.mark.parametrize(
        "value,expected_type,is_optional,expected_outcome",
        [
            (True, bool, False, True),
            ("true", bool, False, True),
            ("false", bool, False, False),
            ("1", bool, False, True),
            ("no", bool, False, False),
            ("not_a_bool", bool, False, ValueError),
            ("100", int, False, 100),
            ("100.0", int, False, 100),
            (100.0, int, False, 100),
            (100.5, int, False, ValueError),
            ("100.5", int, False, ValueError),
            ("not_a_number", int, False, ValueError),
            ("100", float, False, 100.0),
            ("100.5", float, False, 100.5),
            (100, float, False, 100.0),
            ("not_a_number", float, False, ValueError),
            (None, str, True, None),
            ("None", str, True, None),
            ("string_value", str, True, "string_value"),
            (None, str, False, TypeError),
            ("None", str, False, "None"),
        ],
    )
    def test_flexible_type_check(
        self, value, expected_type, is_optional, expected_outcome, parameters
    ):
        if isinstance(expected_outcome, type) and issubclass(
            expected_outcome, Exception
        ):
            with pytest.raises(expected_outcome):
                parameters._flexible_type_check(
                    value, expected_type, is_optional=is_optional
                )
        else:
            result = parameters._flexible_type_check(
                value, expected_type, is_optional=is_optional
            )
            assert result == expected_outcome, (
                f"Expected '{expected_outcome}', got '{result}'"
            )

    @pytest.mark.parametrize(
        "value_range, expected_outcome",
        [
            ((1, 10), (1, 10)),
            ((10, 1), ValueError),  # invalid range, first greater than second
            (("1", 10), ValueError),  # invalid types
            ((1, "10"), ValueError),  # invalid types
            ((1.0, 10.0), (1.0, 10.0)),  # valid range with floats
            ((-10, -1), (-10, -1)),  # valid negative range
            ((0, 0), ValueError),  # invalid range, same numbers
            ((1,), ValueError),  # invalid length
            (None, ValueError),  # None as input
        ],
    )
    def test_validate_range(self, parameters, value_range, expected_outcome):
        if expected_outcome is ValueError:
            with pytest.raises(ValueError):
                parameters._validate_range(value_range)
        else:
            assert parameters._validate_range(value_range) == expected_outcome, (
                f"Expected range {expected_outcome}, got {parameters._validate_range(value_range)}"
            )

    @pytest.mark.parametrize(
        # The "key2" parameter, used in this test, is an integer with a value of 100
        "value_range, expected_outcome",
        [
            ((0, 100), 100),
            ((0, 99), ValueError),
            ((100, 101), 100),
            ((101, 102), ValueError),
        ],
    )
    def test_range_validation_on_value(self, value_range, expected_outcome, parameters):
        if expected_outcome is ValueError:
            with pytest.raises(ValueError):
                parameters.extract_parameter(
                    keys=["key2"], expected_type=int, range_value=value_range
                )
        else:
            assert (
                parameters.extract_parameter(
                    keys=["key2"], expected_type=int, range_value=value_range
                )
                == expected_outcome
            )

    @pytest.mark.parametrize(
        "value,expected_outcome",
        [
            ("CONSTANT", TestTileMode.CONSTANT),
            ("constant", TestTileMode.CONSTANT),
            (0, TestTileMode.CONSTANT),
            ("DROP", TestTileMode.DROP),
            ("drop", TestTileMode.DROP),
            (1, TestTileMode.DROP),
            ("REFLECT", TestTileMode.REFLECT),
            ("reflect", TestTileMode.REFLECT),
            (2, TestTileMode.REFLECT),
            ("EDGE", TestTileMode.EDGE),
            ("edge", TestTileMode.EDGE),
            (3, TestTileMode.EDGE),
            ("WRAP", TestTileMode.WRAP),
            ("wrap", TestTileMode.WRAP),
            (4, TestTileMode.WRAP),
            (TestTileMode.CONSTANT, TestTileMode.CONSTANT),  # Already an enum value
            ("INVALID", ValueError),
            (5, ValueError),
        ],
    )
    def test_extract_parameter_enum(self, parameters, value, expected_outcome):
        parameters.parameters_data["tile_mode"] = value

        if isinstance(expected_outcome, type) and issubclass(
            expected_outcome, Exception
        ):
            with pytest.raises(expected_outcome):
                parameters.extract_parameter(
                    keys=["tile_mode"], expected_type=TestTileMode
                )
        else:
            result = parameters.extract_parameter(
                keys=["tile_mode"], expected_type=TestTileMode
            )
            assert result == expected_outcome, (
                f"Expected {expected_outcome}, got {result}"
            )

    def test_extract_parameter_enum_default(self, parameters):
        result = parameters.extract_parameter(
            keys=["nonexistent"],
            expected_type=TestTileMode,
            default=TestTileMode.CONSTANT,
        )
        assert result == TestTileMode.CONSTANT

    def test_extract_parameter_enum_no_default(self, parameters):
        with pytest.raises(KeyError):
            parameters.extract_parameter(
                keys=["nonexistent"], expected_type=TestTileMode
            )

    def test_extract_parameter_optional_enum(self, parameters):
        parameters.parameters_data["optional_tile_mode"] = None
        result = parameters.extract_parameter(
            keys=["optional_tile_mode"], expected_type=TestTileMode | None
        )
        assert result is None

        parameters.parameters_data["optional_tile_mode"] = "CONSTANT"
        result = parameters.extract_parameter(
            keys=["optional_tile_mode"], expected_type=TestTileMode | None
        )
        assert result == TestTileMode.CONSTANT

    def test_extract_parameter_enum_case_insensitive(self, parameters):
        parameters.parameters_data["tile_mode"] = "cOnStAnT"
        result = parameters.extract_parameter(
            keys=["tile_mode"], expected_type=TestTileMode
        )
        assert result == TestTileMode.CONSTANT

    def test_extract_parameter_enum_invalid_type(self, parameters):
        parameters.parameters_data["tile_mode"] = 3.14  # Float instead of int or string
        with pytest.raises(ValueError):
            parameters.extract_parameter(keys=["tile_mode"], expected_type=TestTileMode)

    def test_optional_parameter(self, parameters):
        assert (
            parameters.extract_parameter(keys=["key5"], expected_type=str | None)
            is None
        )

        assert (
            parameters.extract_parameter(keys=["key6"], expected_type=str | None)
            is None
        )

        assert (
            parameters.extract_parameter(
                keys=["key1"], expected_type=str | None, default=None
            )
            == parameters.parameters_data["key1"]
        )

        assert (
            parameters.extract_parameter(
                keys=["nonexistent"], expected_type=str | None, default=None
            )
            is None
        )

    def test_non_optional_parameter_with_none_parameter_raises_an_error(
        self, parameters
    ):
        with pytest.raises(TypeError):
            parameters.extract_parameter(keys=["key6"], expected_type=str)

    def test_parsed_optional_value(self, parameters):
        assert (
            parameters.extract_parameter(keys=["key5"], expected_type=str)
            == parameters.parameters_data["key5"]
        )

        assert (
            parameters.extract_parameter(keys=["key5"], expected_type=str | None)
            is None
        )

        assert (
            parameters.extract_parameter(
                keys=["nonexistent"], expected_type=str | None, default="null"
            )
            == "null"
        )

    def test_non_optional_parameter_with_none_default(self, parameters):
        parameters.extract_parameter(
            keys=["nonexistent"], expected_type=str | None, default=None
        )
        with pytest.raises(ValueError):
            parameters.extract_parameter(
                keys=["nonexistent"], expected_type=str, default=None
            )
