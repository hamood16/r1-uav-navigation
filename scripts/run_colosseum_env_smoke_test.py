"""Run a small live smoke test for the Colosseum Gymnasium UAV environment."""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np

from r1_uav_nav.envs import ColosseumUAVEnv, ColosseumUAVEnvConfig
from r1_uav_nav.sim import ColosseumClientError

DEFAULT_STEPS = 10
DEFAULT_SEED = 42
DEFAULT_CLIENT_MODULE = "airsim"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse live Colosseum environment smoke-test arguments."""
    parser = argparse.ArgumentParser(
        description="Run a small live smoke test for ColosseumUAVEnv.",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--policy",
        choices=("zero", "forward", "random"),
        default="zero",
    )
    parser.add_argument(
        "--goal-offset",
        default=None,
        help="Optional x,y,z goal offset",
    )
    parser.add_argument(
        "--action",
        default=None,
        help="Optional normalized x,y,z action used instead of --policy",
    )
    parser.add_argument("--client-module", default=DEFAULT_CLIENT_MODULE)
    return parser.parse_args(argv)


def run_smoke_test(args: argparse.Namespace) -> int:
    """Run the smoke test and return a process-style exit code."""
    if args.steps < 0:
        raise ValueError("steps must be non-negative")

    env = ColosseumUAVEnv(
        ColosseumUAVEnvConfig(client_module=args.client_module),
    )
    rng = np.random.default_rng(args.seed)
    explicit_action = (
        _parse_triplet(args.action, "action") if args.action is not None else None
    )
    options = {}
    if args.goal_offset is not None:
        options["goal_offset"] = _parse_triplet(args.goal_offset, "goal-offset")

    operation_failed = False
    try:
        _observation, info = env.reset(seed=args.seed, options=options)
        print("Reset complete.")
        _print_step_summary(
            step_index=0,
            action=None,
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info,
        )

        for step_index in range(1, args.steps + 1):
            action = _select_action(args.policy, explicit_action, rng)
            _observation, reward, terminated, truncated, info = env.step(action)
            _print_step_summary(
                step_index=step_index,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            if terminated or truncated:
                break
    except (ColosseumClientError, RuntimeError, ValueError) as exc:
        operation_failed = True
        print(f"Colosseum environment smoke test failed: {exc}")
    except KeyboardInterrupt:
        operation_failed = True
        print("Colosseum environment smoke test interrupted by user.")
    finally:
        env.close()
        if env.last_cleanup_result is not None:
            _print_cleanup_result(env.last_cleanup_result)

    cleanup_failed = (
        env.last_cleanup_result is not None
        and env.last_cleanup_result.safety_critical_failure
    )
    if cleanup_failed:
        print("Colosseum environment smoke test failed during cleanup.")

    if operation_failed or cleanup_failed:
        return 1
    print("Colosseum environment smoke test complete.")
    return 0


def main() -> int:
    """Run the live smoke test."""
    return run_smoke_test(parse_args())


def _select_action(
    policy: str,
    explicit_action: tuple[float, float, float] | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if explicit_action is not None:
        return np.asarray(explicit_action, dtype=np.float32)
    if policy == "zero":
        return np.zeros(3, dtype=np.float32)
    if policy == "forward":
        return np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
    if policy == "random":
        return rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
    raise ValueError(f"Unknown policy: {policy}")


def _parse_triplet(value: str, label: str) -> tuple[float, float, float]:
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError(f"{label} must be formatted as x,y,z")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _print_step_summary(
    *,
    step_index: int,
    action: np.ndarray | None,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, object],
) -> None:
    action_text = "reset" if action is None else np.array2string(action, precision=3)
    print(
        f"step={step_index} action={action_text} "
        f"position={info['measured_position']} "
        f"distance={float(info['distance_to_goal']):.3f} "
        f"reward={reward:.3f} terminated={terminated} truncated={truncated} "
        f"reason={info['termination_reason']}"
    )


def _print_cleanup_result(cleanup_result: object) -> None:
    actions_succeeded = getattr(cleanup_result, "actions_succeeded", ())
    errors = getattr(cleanup_result, "errors", ())
    if actions_succeeded:
        print("Cleanup actions:")
        for action in actions_succeeded:
            print(f"- {action}")
    if errors:
        print("Cleanup errors:")
        for error in errors:
            print(f"- {error}")


if __name__ == "__main__":
    raise SystemExit(main())
