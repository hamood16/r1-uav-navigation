"""Check a basic AirSim-style Colosseum multirotor connection."""

from __future__ import annotations

import argparse
from typing import Sequence

from r1_uav_nav.sim import (
    ColosseumClientError,
    ColosseumClientImportError,
    confirm_connection,
    create_multirotor_client,
    get_multirotor_state_summary,
    import_colosseum_client_module,
    perform_basic_control_check,
    read_multirotor_state,
)

DEFAULT_CLIENT_MODULE = "airsim"
DEFAULT_DURATION = 2.0
DEFAULT_VELOCITY = 1.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse Colosseum connection check arguments."""
    parser = argparse.ArgumentParser(
        description="Check an AirSim-style Colosseum multirotor connection.",
    )
    parser.add_argument("--client-module", default=DEFAULT_CLIENT_MODULE)
    parser.add_argument("--takeoff", action="store_true")
    parser.add_argument("--move-demo", action="store_true")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--velocity", type=float, default=DEFAULT_VELOCITY)
    return parser.parse_args(argv)


def main() -> int:
    """Run the connection check."""
    args = parse_args()

    try:
        client_module = import_colosseum_client_module(args.client_module)
        client = create_multirotor_client(client_module)
        confirm_connection(client)
        state = read_multirotor_state(client)
        summary = get_multirotor_state_summary(state)
        actions = perform_basic_control_check(
            client,
            takeoff=args.takeoff,
            move_demo=args.move_demo,
            duration=args.duration,
            velocity=args.velocity,
        )
    except (ColosseumClientError, ColosseumClientImportError, ValueError) as exc:
        print(f"Colosseum connection check failed: {exc}")
        return 1

    print("Colosseum connection check succeeded.")
    print("Multirotor state:")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    if actions:
        print("Control actions:")
        for action in actions:
            print(f"- {action}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
