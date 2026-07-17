# M12 Colosseum Setup

## Purpose

M12.2 adds the first local Colosseum connection workflow for `r1-UAV-navigation`.
The goal is to help a user manually start a Colosseum-compatible Unreal UAV
environment, connect from Python, read drone state, and optionally run a basic
takeoff or short movement check.

This milestone is a simulator connection step only. It does not add 3D
reinforcement-learning training, a Gymnasium wrapper, camera-based RL, or PX4-first
integration.

## Manual Local Setup Checklist

Validated Windows setup:

- Prebuilt simulator: Colosseum Blocks, version `v2.0.0-beta`.
- Matching Colosseum repository tag: `v2.0.0-beta`.
- Matching Colosseum repository commit: `7b9658a1`.
- Python client install mode: editable install from the matching
  `Colosseum/PythonClient` directory.
- Validated RPC compatibility pin: `msgpack==0.6.2`.

The simulator binary, the Colosseum repository checkout, and the virtual
environment are not committed to this repository.

1. Obtain the prebuilt Colosseum Blocks simulator for `v2.0.0-beta`, or build a
   Colosseum-compatible Unreal environment from the same tag.
2. Check out the matching Colosseum repository tag and commit:

   ```powershell
   git clone https://github.com/CodexLabsLLC/Colosseum.git C:\Colosseum
   cd C:\Colosseum
   git fetch --tags
   git checkout v2.0.0-beta
   git rev-parse --short HEAD
   ```

   The final command should report `7b9658a1`.

3. Activate the project Python virtual environment:

   ```powershell
   cd C:\Hamdhan\Projects\r1-UAV-navigation
   .\.venv\Scripts\Activate.ps1
   ```

4. Install the matching Colosseum Python client from the checked-out repository:

   ```powershell
   python -m pip install -e C:\Colosseum\PythonClient
   python -m pip install "msgpack==0.6.2"
   ```

5. Start the Blocks simulator manually on the local machine. For example, from
   the extracted prebuilt Blocks folder:

   ```powershell
   .\Blocks.exe
   ```

6. Ensure the simulator is configured for multirotor or drone mode.
7. Run the read-only connection check script:

   ```powershell
   python scripts\check_colosseum_connection.py
   ```

Colosseum and Unreal binaries are not committed to this repository. Keep simulator
builds, downloaded environments, logs, and generated assets outside version control.

## Python Client Notes

Colosseum follows an AirSim-style Python workflow. Depending on the Colosseum
release and setup method, the Python API may be imported as `airsim`.

The project does not add `airsim` or Colosseum as a core dependency in M12.2 because
the correct client install path may depend on the chosen Colosseum release. The
connection script fails gracefully if the Python client module is missing.

For the validated Blocks setup, keep the simulator binary and Python client aligned:

- Blocks simulator: Colosseum `v2.0.0-beta`
- Colosseum repository: tag `v2.0.0-beta`, commit `7b9658a1`
- Python client: editable install from `C:\Colosseum\PythonClient`

`msgpack-rpc-python==0.4.1` is incompatible with `msgpack` 1.x because it passes the
removed `encoding` argument. Use `msgpack==0.6.2` for the validated setup.

If your client module has a different import name, pass it explicitly:

```powershell
python scripts\check_colosseum_connection.py --client-module airsim
```

## Connection Check Commands

Read drone state only:

```powershell
python scripts\check_colosseum_connection.py
```

This prints a compact state summary with landed state, position, and linear
velocity.

Run a basic takeoff check:

```powershell
python scripts\check_colosseum_connection.py --takeoff
```

Run takeoff plus a short forward movement demo:

```powershell
python scripts\check_colosseum_connection.py --takeoff --move-demo
```

The default command only connects and reads drone state. Takeoff and movement are
always opt-in.

## Troubleshooting

| Symptom | Possible cause | Suggested check |
|---|---|---|
| Python client module missing | Client package is not installed or not on `PYTHONPATH` | Install or expose the Colosseum/AirSim-compatible Python client for the active `.venv`. |
| Connection refused | Simulator is not running or not listening | Start the Unreal simulator before running the Python script. |
| Drone state cannot be read | Drone is not spawned or API mode is unavailable | Confirm the scene contains a multirotor vehicle and API control is enabled. |
| Commands do nothing | Wrong simulator mode | Ensure the simulator is in multirotor/drone mode, not car or computer-vision-only mode. |
| Intermittent connection failure | Firewall or port issue | Allow the simulator through the firewall and confirm the local API port is reachable. |
| Takeoff fails | Vehicle is not ready or not armed | Try the read-only connection check first, then retry `--takeoff`. |
| `TypeError: unexpected keyword argument 'encoding'` | `msgpack-rpc-python==0.4.1` is being used with `msgpack` 1.x | Install the validated pin with `python -m pip install "msgpack==0.6.2"`. |
| `ValueError: Length of encoded data does not match number of attributes` | Python client and simulator binary are likely from different Colosseum versions | Check out Colosseum tag `v2.0.0-beta` at commit `7b9658a1` and reinstall from `C:\Colosseum\PythonClient`. |

## M12.2 Success Criteria

M12.2 is successful when:

- the Colosseum simulator opens locally
- the Python client imports
- the Python client connects to the simulator
- drone state can be read from Python
- optional takeoff or short movement commands work when explicitly requested

## M12.2 Non-Goals

- no camera-based RL
- no Gymnasium wrapper
- no RL training
- no RL evaluation
- no PX4-first integration
- no simulator binaries committed to the repo
- no generated artifacts committed to the repo
