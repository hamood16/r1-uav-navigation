from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from r1_uav_nav.sim import (
    ColosseumClientError,
    ColosseumClientImportError,
    create_multirotor_client,
    get_multirotor_state_summary,
    import_colosseum_client_module,
    perform_basic_control_check,
)

DOC_PATH = Path("docs/m12_colosseum_setup.md")


class FakeAsyncResult:
    def __init__(self, client: "FakeClient", action: str) -> None:
        self.client = client
        self.action = action

    def join(self) -> None:
        self.client.calls.append(f"{self.action}.join")


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def enableApiControl(self, enabled: bool) -> None:
        self.calls.append(("enableApiControl", enabled))

    def armDisarm(self, armed: bool) -> None:
        self.calls.append(("armDisarm", armed))

    def takeoffAsync(self) -> FakeAsyncResult:
        self.calls.append("takeoffAsync")
        return FakeAsyncResult(self, "takeoffAsync")

    def moveByVelocityAsync(
        self,
        vx: float,
        vy: float,
        vz: float,
        duration: float,
    ) -> FakeAsyncResult:
        self.calls.append(("moveByVelocityAsync", vx, vy, vz, duration))
        return FakeAsyncResult(self, "moveByVelocityAsync")

    def hoverAsync(self) -> FakeAsyncResult:
        self.calls.append("hoverAsync")
        return FakeAsyncResult(self, "hoverAsync")

    def landAsync(self) -> FakeAsyncResult:
        self.calls.append("landAsync")
        return FakeAsyncResult(self, "landAsync")


class FakeFailingMoveClient(FakeClient):
    def moveByVelocityAsync(
        self,
        vx: float,
        vy: float,
        vz: float,
        duration: float,
    ) -> FakeAsyncResult:
        self.calls.append(("moveByVelocityAsync", vx, vy, vz, duration))
        raise RuntimeError("simulator rejected movement")


def test_m12_colosseum_setup_doc_exists() -> None:
    assert DOC_PATH.exists()


def test_m12_colosseum_setup_doc_mentions_required_topics() -> None:
    doc_text = DOC_PATH.read_text(encoding="utf-8")

    assert "Colosseum" in doc_text
    assert "Python client" in doc_text
    assert "drone state" in doc_text
    assert "takeoff" in doc_text
    assert "no camera-based RL" in doc_text


def test_m12_colosseum_setup_doc_records_validated_windows_setup() -> None:
    doc_text = DOC_PATH.read_text(encoding="utf-8")

    assert "v2.0.0-beta" in doc_text
    assert "7b9658a1" in doc_text
    assert "Colosseum/PythonClient" in doc_text
    assert "msgpack==0.6.2" in doc_text
    assert "unexpected keyword argument 'encoding'" in doc_text
    assert "Length of encoded data does not match number of attributes" in doc_text


def test_check_colosseum_connection_parse_args_defaults() -> None:
    module = _load_connection_script_module()

    args = module.parse_args([])

    assert args.client_module == "airsim"
    assert args.takeoff is False
    assert args.move_demo is False
    assert args.duration == pytest.approx(2.0)
    assert args.velocity == pytest.approx(1.0)


def test_check_colosseum_connection_parse_args_custom_flags() -> None:
    module = _load_connection_script_module()

    args = module.parse_args(
        [
            "--client-module",
            "custom_airsim",
            "--takeoff",
            "--move-demo",
            "--duration",
            "3.5",
            "--velocity",
            "1.25",
        ]
    )

    assert args.client_module == "custom_airsim"
    assert args.takeoff is True
    assert args.move_demo is True
    assert args.duration == pytest.approx(3.5)
    assert args.velocity == pytest.approx(1.25)


def test_missing_client_module_raises_clear_error() -> None:
    with pytest.raises(ColosseumClientImportError, match="Could not import"):
        import_colosseum_client_module("definitely_missing_colosseum_client")


def test_create_multirotor_client_uses_injected_module() -> None:
    module = SimpleNamespace(MultirotorClient=FakeClient)

    client = create_multirotor_client(module)  # type: ignore[arg-type]

    assert isinstance(client, FakeClient)


def test_state_summary_formats_fake_state_object() -> None:
    state = SimpleNamespace(
        landed_state=1,
        kinematics_estimated=SimpleNamespace(
            position=SimpleNamespace(x_val=1.0, y_val=2.0, z_val=-3.0),
            linear_velocity=SimpleNamespace(x_val=0.5, y_val=0.0, z_val=-0.1),
        ),
    )

    summary = get_multirotor_state_summary(state)

    assert summary == {
        "landed_state": "1",
        "position": "x=1.000, y=2.000, z=-3.000",
        "linear_velocity": "x=0.500, y=0.000, z=-0.100",
    }


def test_basic_control_check_does_nothing_without_control_flags() -> None:
    client = FakeClient()

    actions = perform_basic_control_check(
        client,
        takeoff=False,
        move_demo=False,
        duration=2.0,
        velocity=1.0,
    )

    assert actions == []
    assert client.calls == []


def test_basic_control_check_runs_takeoff_and_move_demo_with_fake_client() -> None:
    client = FakeClient()

    actions = perform_basic_control_check(
        client,
        takeoff=True,
        move_demo=True,
        duration=2.0,
        velocity=1.0,
    )

    assert actions == [
        "enabled API control",
        "armed drone",
        "completed takeoff",
        "completed forward velocity demo",
        "completed hover",
        "completed landing",
        "disarmed drone",
        "disabled API control",
    ]
    assert client.calls == [
        ("enableApiControl", True),
        ("armDisarm", True),
        "takeoffAsync",
        "takeoffAsync.join",
        ("moveByVelocityAsync", 1.0, 0.0, 0.0, 2.0),
        "moveByVelocityAsync.join",
        "hoverAsync",
        "hoverAsync.join",
        "landAsync",
        "landAsync.join",
        ("armDisarm", False),
        ("enableApiControl", False),
    ]


def test_basic_control_check_cleans_up_after_takeoff_only() -> None:
    client = FakeClient()

    actions = perform_basic_control_check(
        client,
        takeoff=True,
        move_demo=False,
        duration=2.0,
        velocity=1.0,
    )

    assert actions == [
        "enabled API control",
        "armed drone",
        "completed takeoff",
        "completed hover",
        "completed landing",
        "disarmed drone",
        "disabled API control",
    ]
    assert client.calls == [
        ("enableApiControl", True),
        ("armDisarm", True),
        "takeoffAsync",
        "takeoffAsync.join",
        "hoverAsync",
        "hoverAsync.join",
        "landAsync",
        "landAsync.join",
        ("armDisarm", False),
        ("enableApiControl", False),
    ]


def test_basic_control_check_preserves_move_error_after_cleanup() -> None:
    client = FakeFailingMoveClient()

    with pytest.raises(
        ColosseumClientError,
        match="moveByVelocityAsync",
    ):
        perform_basic_control_check(
            client,
            takeoff=True,
            move_demo=True,
            duration=2.0,
            velocity=1.0,
        )

    assert client.calls == [
        ("enableApiControl", True),
        ("armDisarm", True),
        "takeoffAsync",
        "takeoffAsync.join",
        ("moveByVelocityAsync", 1.0, 0.0, 0.0, 2.0),
        "hoverAsync",
        "hoverAsync.join",
        "landAsync",
        "landAsync.join",
        ("armDisarm", False),
        ("enableApiControl", False),
    ]


def _load_connection_script_module() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "check_colosseum_connection.py"
    )
    spec = spec_from_file_location("check_colosseum_connection", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
