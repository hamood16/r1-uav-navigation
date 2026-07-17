"""Small helpers for AirSim-style Colosseum client checks."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

DEFAULT_CLIENT_MODULE = "airsim"


class ColosseumClientError(RuntimeError):
    """Raised when a Colosseum/AirSim-style client operation fails."""


class ColosseumClientImportError(ColosseumClientError):
    """Raised when the Colosseum/AirSim-style Python client cannot be imported."""


def import_colosseum_client_module(
    module_name: str = DEFAULT_CLIENT_MODULE,
) -> ModuleType:
    """Import the Colosseum/AirSim-compatible Python client module."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ColosseumClientImportError(
            f"Could not import Python client module {module_name!r}. Install or expose "
            "the Colosseum/AirSim-compatible client in the active environment."
        ) from exc


def create_multirotor_client(client_module: ModuleType) -> Any:
    """Create an AirSim-style MultirotorClient from a client module."""
    try:
        client_class = client_module.MultirotorClient
    except AttributeError as exc:
        raise ColosseumClientError(
            "Client module does not provide MultirotorClient. Confirm that the "
            "selected module is an AirSim-style multirotor client."
        ) from exc

    return client_class()


def confirm_connection(client: Any) -> None:
    """Confirm the simulator connection using the AirSim-style client."""
    try:
        client.confirmConnection()
    except Exception as exc:
        raise ColosseumClientError(
            "Could not confirm simulator connection. Make sure Colosseum is running "
            "and listening for Python API connections."
        ) from exc


def read_multirotor_state(client: Any) -> Any:
    """Read multirotor state from the simulator client."""
    try:
        return client.getMultirotorState()
    except Exception as exc:
        raise ColosseumClientError(
            "Could not read multirotor state. Confirm that a drone is spawned and "
            "the simulator is in multirotor mode."
        ) from exc


def get_multirotor_state_summary(state: Any) -> dict[str, str]:
    """Return a compact, printable summary for an AirSim-style multirotor state."""
    kinematics = getattr(state, "kinematics_estimated", None)
    position = getattr(kinematics, "position", None)
    linear_velocity = getattr(kinematics, "linear_velocity", None)
    landed_state = getattr(state, "landed_state", "unknown")

    return {
        "landed_state": str(landed_state),
        "position": _format_vector(position),
        "linear_velocity": _format_vector(linear_velocity),
    }


def perform_basic_control_check(
    client: Any,
    *,
    takeoff: bool,
    move_demo: bool,
    duration: float,
    velocity: float,
) -> list[str]:
    """Run optional, explicit AirSim-style takeoff and movement checks."""
    if not takeoff and not move_demo:
        return []

    _require_positive("duration", duration)
    _require_positive("velocity", velocity)

    actions: list[str] = []
    cleanup_required = False

    try:
        _call_client_method(client, "enableApiControl", True)
        actions.append("enabled API control")
        cleanup_required = True

        _call_client_method(client, "armDisarm", True)
        actions.append("armed drone")

        _wait_for_async_result(_call_client_method(client, "takeoffAsync"))
        actions.append("completed takeoff")

        if move_demo:
            _wait_for_async_result(
                _call_client_method(
                    client,
                    "moveByVelocityAsync",
                    velocity,
                    0.0,
                    0.0,
                    duration,
                )
            )
            actions.append("completed forward velocity demo")
    finally:
        if cleanup_required:
            actions.extend(_cleanup_after_control(client))

    return actions


def _format_vector(vector: Any) -> str:
    if vector is None:
        return "unknown"

    try:
        x_value = float(vector.x_val)
        y_value = float(vector.y_val)
        z_value = float(vector.z_val)
    except (AttributeError, TypeError, ValueError):
        return "unknown"

    return f"x={x_value:.3f}, y={y_value:.3f}, z={z_value:.3f}"


def _call_client_method(client: Any, method_name: str, *args: Any) -> Any:
    method = getattr(client, method_name, None)
    if method is None:
        raise ColosseumClientError(
            f"Client does not provide required method {method_name!r}."
        )

    try:
        return method(*args)
    except Exception as exc:
        raise ColosseumClientError(
            f"Client method {method_name!r} failed during the basic control check."
        ) from exc


def _wait_for_async_result(async_result: Any) -> None:
    join_method = getattr(async_result, "join", None)
    if join_method is None:
        return

    join_method()


def _cleanup_after_control(client: Any) -> list[str]:
    actions: list[str] = []

    if _call_optional_async_cleanup(client, "hoverAsync"):
        actions.append("completed hover")
    if _call_optional_async_cleanup(client, "landAsync"):
        actions.append("completed landing")
    if _call_optional_cleanup(client, "armDisarm", False):
        actions.append("disarmed drone")
    if _call_optional_cleanup(client, "enableApiControl", False):
        actions.append("disabled API control")

    return actions


def _call_optional_async_cleanup(client: Any, method_name: str) -> bool:
    method = getattr(client, method_name, None)
    if method is None:
        return False

    try:
        _wait_for_async_result(method())
    except Exception:
        return False

    return True


def _call_optional_cleanup(client: Any, method_name: str, *args: Any) -> bool:
    method = getattr(client, method_name, None)
    if method is None:
        return False

    try:
        method(*args)
    except Exception:
        return False

    return True


def _require_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
