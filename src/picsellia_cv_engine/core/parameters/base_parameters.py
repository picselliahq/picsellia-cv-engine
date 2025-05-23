import logging
from abc import ABC
from enum import Enum
from typing import (
    Any,
    Generic,
    TypeVar,
    Union,
    get_args,
    get_origin,
    overload,
)

from picsellia.types.schemas import LogDataType  # type: ignore

from picsellia_cv_engine.core import Colors

logger = logging.getLogger("picsellia-engine")
T = TypeVar("T")


class Parameters(ABC, Generic[T]):
    """
    Base class for handling typed parameter extraction from Picsellia log data.
    """

    def __init__(self, log_data: LogDataType):
        """
        Initialize with log data.

        Args:
            log_data (LogDataType): Dictionary of parameters extracted from Picsellia logs.

        Raises:
            ValueError: If the provided data is not a dictionary.
        """
        self.parameters_data = self.validate_log_data(log_data)

        # Store the keys that have been defaulted, used for logging purposes
        self.defaulted_keys: set[str] = set()

    @overload
    def extract_parameter(
        self,
        keys: list,
        expected_type: type[T],
        default: Any = ...,
        range_value: tuple[Any, Any] | None = None,
    ) -> T: ...

    @overload
    def extract_parameter(
        self,
        keys: list,
        expected_type: Any,
        default: Any = ...,
        range_value: tuple[Any, Any] | None = None,
    ) -> Any: ...

    def extract_parameter(
        self,
        keys: list,
        expected_type: type[T],
        default: Any = ...,
        range_value: tuple[Any, Any] | None = None,
    ) -> Any:
        """
        Extract a parameter using keys, type, optional default, and optional value range.

        Examples:
            Extract a required string parameter that cannot be None:
            ```
            parameter = self.extract_parameter(keys=["key1", "key2"], expected_type=str)
            ```

            Extract a required integer parameter that can be None:
            ```
            parameter = self.extract_parameter(keys=["key1"], expected_type=int | None)
            ```

            Extract an optional float parameter within a specific range:
            ```
            parameter = self.extract_parameter(keys=["key1"], expected_type=float, default=0.5, range_value=(0.0, 1.0))
            ```

            Extract an optional string parameter with a default value:
            ```
            parameter = self.extract_parameter(keys=["key1"], expected_type=str, default="default_value")
            ```

            Extract an optional string parameter that can be None:
            ```
            parameter = self.extract_parameter(keys=["key1"], expected_type=Union[str, None], default=None)
            ```

        Args:
            keys: A list of possible keys to extract the parameter.
            expected_type: The expected type of the parameter, can use Union for optional types.
            default: The default value if the parameter is not found. Use ... for required parameters.
            range_value: A tuple of two numbers representing the allowed range of the parameter.

        Returns:
            The parsed parameter.

        Raises:
            ValueError: If no keys are provided or if the value is out of the allowed range.
            TypeError: If the parameter is not of the expected type.
            KeyError: If no parameter is found and no default value is provided.
        """

        if not keys:
            raise ValueError(
                "Cannot extract a parameter without any keys. One or more keys must be provided."
            )

        # Determine if the type is optional
        origin = get_origin(expected_type)
        args = get_args(expected_type)
        is_optional = origin is Union and any(isinstance(None, arg) for arg in args)
        base_type = (
            next((arg for arg in args if not isinstance(None, arg)), expected_type)
            if is_optional
            else expected_type
        )

        self._validate_default_value(default, base_type, is_optional, expected_type)

        for key in keys:
            if key in self.parameters_data:
                return self._process_parameter_value(
                    key, expected_type, base_type, range_value, is_optional
                )

        return self._handle_missing_parameter(keys, expected_type, default, range_value)

    def _validate_default_value(self, default, base_type, is_optional, expected_type):
        # Check if the default value matches the expected type
        if default is not ... and not isinstance(default, base_type | type(None)):
            raise TypeError(
                f"The provided default value {default} does not match the expected type {expected_type}."
            )

        if default is None and not is_optional:
            raise ValueError(
                f"The default value cannot be None as the expected type {expected_type} is not optional."
            )

    def _process_parameter_value(
        self, key, expected_type, base_type, range_value, is_optional
    ):
        value = self.parameters_data[key]
        parsed_value = self._flexible_type_check(
            value, base_type, is_optional=is_optional
        )

        if parsed_value is None and not is_optional:
            raise TypeError(
                f"The value for key '{key}' cannot be None as it's not an optional type."
            )

        if parsed_value is not None:
            if isinstance(expected_type, type) and issubclass(expected_type, Enum):
                try:
                    return expected_type(parsed_value)
                except ValueError:
                    raise ValueError(
                        f"Invalid value '{parsed_value}' for enum {expected_type.__name__}"
                    ) from None

            if range_value and base_type in [int, float]:
                checked_value_range = self._validate_range(range_value)
                if not (
                    checked_value_range[0] <= parsed_value <= checked_value_range[1]
                ):
                    raise ValueError(
                        f"Value for key '{key}' is out of the allowed range {range_value}."
                    )
            return parsed_value

        elif is_optional:
            return parsed_value

        else:
            raise RuntimeError(
                f"The value {value} for key {key} has been parsed to None and therefore cannot be used. "
                f"The key {key} expects a value of type {expected_type}."
            )

    def _handle_missing_parameter(self, keys, expected_type, default, range_value):
        if default is not ...:
            logger.warning(
                f"None of the keys {keys} were found in the provided data. "
                f"Using default value {Colors.YELLOW}{default}{Colors.ENDC}."
            )
            self.defaulted_keys.update(keys)
            return default

        else:
            error_string = f"Required parameter with key(s) {keys} of type {expected_type} not found."

            if range_value is not None:
                error_string += f" Expected value within the range {range_value}."

            raise KeyError(error_string)

    def to_dict(self) -> dict[str, Any]:
        """Return parameters as a dictionary, excluding internal fields."""
        filtered_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ["parameters_data", "defaulted_keys"]
        }
        return dict(sorted(filtered_dict.items()))

    def validate_log_data(self, log_data: LogDataType) -> dict[str, Any]:
        """Validate and return log data if it's a dictionary."""
        if isinstance(log_data, dict):
            return log_data

        raise ValueError("The provided parameters must be a dictionary.")

    def _flexible_type_check(
        self, value: Any, expected_type: type[T], is_optional: bool
    ) -> Any:
        """Try to cast value to expected type, handling optional and coercion logic."""
        if expected_type is bool:
            return self._check_bool(value)

        elif expected_type is float:
            return self._check_float(value)

        elif expected_type is int:
            return self._check_int(value)

        elif isinstance(expected_type, type) and issubclass(expected_type, Enum):
            return self._check_enum(value, expected_type, is_optional)

        elif value is None and not is_optional:
            raise TypeError(
                f"Value {value} cannot be None as it's not an optional type."
            )

        elif is_optional:
            return self._check_optional(value)

        return value

    def _check_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if str(value).lower() in ["1", "true", "yes"]:
            return True
        if str(value).lower() in ["0", "false", "no"]:
            return False

        raise ValueError(
            f"Value {value} cannot be converted to a boolean."
            f"Accepted values are '1', 'true', 'yes', '0', 'false', 'no'."
        )

    def _check_float(self, value: Any) -> float:
        if isinstance(value, int | float):
            return float(value)  # Directly converts int to float or maintains float
        try:
            return float(str(value))  # Attempt to convert string to float
        except ValueError as e:
            raise ValueError(f"Value {value} cannot be converted to float.") from e

    def _check_int(self, value: Any) -> int:
        if isinstance(value, int):
            return value  # No conversion needed
        elif isinstance(value, float):
            if value.is_integer():
                return int(value)  # Convert to int if it's a whole number
            else:
                raise ValueError(
                    f"Value {value} cannot be converted to int without losing precision."
                )
        else:
            try:
                # Attempt to convert string to float first to handle cases like "100.0"
                float_value = float(str(value))
                if float_value.is_integer():
                    return int(float_value)  # Convert to int if it's a whole number
                else:
                    raise ValueError
            except ValueError as e:
                raise ValueError(
                    f"Value {value} cannot be converted to int without losing precision."
                ) from e

    def _check_enum(self, value: Any, expected_type, is_optional: bool) -> T | None:
        if isinstance(value, expected_type):
            return value

        elif isinstance(value, str):
            try:
                return expected_type[value.upper()]
            except KeyError:
                try:
                    return expected_type(value.lower())
                except ValueError:
                    pass

        elif isinstance(value, int):
            try:
                return expected_type(value)
            except ValueError:
                pass

        elif value is None and is_optional:
            return None

        valid_values = ", ".join([f"{e.name}({e.value})" for e in expected_type])
        raise ValueError(
            f"Invalid value '{value}' for enum {expected_type.__name__}. "
            f"Valid values are: {valid_values}"
        )

    def _check_optional(self, value: Any) -> Any:
        if value is None:
            return value
        elif str(value).lower() in ["none", "null"]:
            return None

    def _validate_range(self, value_range: tuple) -> tuple:
        """Ensure range is valid and return it."""
        if value_range is not None:
            if (
                len(value_range) == 2
                and isinstance(value_range[0], int | float)
                and isinstance(value_range[1], int | float)
                and value_range[0] < value_range[1]
            ):
                return value_range

        raise ValueError(
            f"The provided range value {value_range} is invalid. "
            "It must be a tuple of two numbers (int or float) "
            "where the first is strictly less than the second."
        )


TParameters = TypeVar("TParameters", bound=Parameters)
