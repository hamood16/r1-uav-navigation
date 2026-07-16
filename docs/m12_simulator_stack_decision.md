# M12 Simulator Stack Decision

## Current Project Status

`r1-UAV-navigation` has completed the core 2D reinforcement-learning foundation:

- Static and dynamic 2D Gymnasium UAV environments are implemented and tested.
- Dynamic DQN baselines and tuned runs exist for `DynamicGridUAVEnv`.
- Continuous TD3 baselines and tuned runs exist for `ContinuousDynamicUAVEnv`.
- Repeated evaluation tooling records mean and population standard deviation across seed blocks.
- Tuned dynamic RL results are documented in `docs/m10_dynamic_rl_results.md`.
- A project-level DQN vs TD3 comparison is documented in `docs/results/dqn_vs_td3_dynamic.md`.

M12 moves the project carefully from local 2D Gymnasium environments toward 3D Unreal-based UAV simulation.

## Decision

Use **Colosseum** as the primary 3D simulator stack for M12.

Colosseum is the preferred next step because it keeps the project close to the AirSim-style Python control workflow while providing an Unreal-based UAV simulation path. It also leaves room for later PX4 or ArduPilot workflows without forcing that complexity into the first 3D milestone.

## Alternatives Considered

| Option | Fit for M12 | Notes |
|---|---|---|
| Microsoft AirSim | Possible but less preferred | AirSim established the Python workflow this project wants, but active ecosystem and maintenance choices should be handled carefully. |
| Colosseum | Chosen | Unreal-based, AirSim-style, UAV-oriented, and suitable for a staged transition from 2D Gymnasium work. |
| Unreal + PX4/Gazebo-style stack | Later-stage option | More realistic, but too heavy for the first simulator connection milestone. |

## Why Colosseum

- It supports an AirSim-style Python control workflow.
- It is Unreal-based, matching the project's planned 3D direction.
- It is a suitable staged next step after the current 2D Gymnasium environments.
- It can support PX4 or ArduPilot workflows later if the project needs higher-fidelity autonomy.
- It has lower immediate complexity than a full PX4-first setup.

## Why Not Full PX4 Immediately

A PX4-first setup is attractive for realism, but it adds flight-controller, firmware, networking, simulator bridge, and configuration complexity before the project has proven a basic 3D simulator loop.

M12 should first prove that the simulator can open, the Python client can connect, drone state can be read, and a basic control command can be executed. PX4 integration can be revisited after that foundation exists.

## M12 Staged Roadmap

| Stage | Goal |
|---|---|
| M12.1 | Choose and document the simulator stack. |
| M12.2 | Run a basic Colosseum simulator instance and connect with a Python client. |
| M12.3 | Build a manual or random 3D navigation demo with simple target-point logging. |
| M12.4 | Add a Gymnasium-style Colosseum UAV wrapper. |
| M12.5 | Train and evaluate a basic 3D baseline. |

## Initial 3D Environment Scope

The first 3D environment should stay intentionally small:

- no camera-based RL at first
- no complex perception pipeline
- state-vector observations only
- simple target-reaching task
- simple collision signal
- simple reward based on reaching the goal, avoiding collision, and making progress

Recommended initial observation fields:

- drone position
- drone velocity
- goal vector
- distance-to-goal
- collision flag

## Initial Action-Space Recommendation

Use continuous velocity commands `vx`, `vy`, `vz` as the first 3D action space.

TD3, SAC, or PPO are more natural choices for this continuous-control setup. DQN should only be used if the 3D action space is deliberately discretised into a small set of movement commands.

## Hardware And Software Assumptions

- Development machine: Windows 11.
- GPU: NVIDIA RTX 4050 laptop GPU available.
- Unreal and Colosseum may be heavy, so precompiled binaries should be tested first if available.
- The Colosseum Python client should be isolated from the core 2D code initially.
- Simulator wrapper tests should mock the simulator client where possible.

## M12.2 Acceptance Criteria

M12.2 should be considered successful when:

- the simulator opens locally
- a Python client connects to the simulator
- drone state can be read from Python
- the drone can arm and take off, or execute one basic movement command
- setup notes and any required local steps are documented

M12.2 should not train models, implement a full Gymnasium wrapper, or start camera-based perception RL.

## Risks And Cautions

- Do not jump straight into PX4 complexity.
- Do not start with camera-based RL or perception-heavy observations.
- First prove simulator connection and simple control.
- Keep Colosseum integration isolated from the existing 2D environments until the simulator workflow is stable.
- Avoid adding simulator dependencies to the core package until the install and runtime story is understood.
