# M12 Colosseum Waypoint Navigation Demo

## Purpose

M12.3 adds scripted 3D waypoint-navigation demonstrations for the validated
Colosseum Blocks setup. These manoeuvres show deterministic simulator control with
absolute waypoint commands. They are not reinforcement-learning navigation, do not
create a Gymnasium wrapper, and do not train or load any policy.

The demo supports three routes:

- `horizontal-square`: a simple square in the horizontal x-y plane.
- `figure-eight`: a Gerono-style figure-eight in the horizontal x-y plane.
- `vertical-square`: a small square in the x-z plane while y remains fixed.

## Coordinate Convention

Colosseum follows the AirSim-style NED coordinate convention where airborne z values
are negative relative to the local reference frame. A negative waypoint `dz` offset
therefore means upward movement.

The script does not assume the simulator starts at `z=0`. It first reads the
measured initial position and preserves:

```text
ground_reference_z = initial_position.z
target_anchor_z = initial_position.z - anchor_altitude
```

Ground clearance is validated with:

```text
clearance = ground_reference_z - absolute_waypoint.z
```

Every route is generated as relative offsets from the measured post-takeoff anchor,
so each manoeuvre begins and ends near the same actual airborne position.

## Safety Defaults

- Default route: `horizontal-square`
- Default anchor altitude magnitude: `2.0`
- Default velocity: `0.5`
- Default waypoint tolerance: `0.5`
- Default waypoint timeout: `20` seconds
- Default figure-eight x/y scale: `3.0` by `2.0`
- Default figure-eight samples: `13`
- Default minimum ground clearance: `1.0`
- Default hover pause between suite routes: `1.0`

The script validates route dimensions, waypoint tolerance, movement velocity,
figure-eight sample count, and ground clearance before arming the UAV. Cleanup is
state-aware and attempts hover, land, disarm, and API-control release for the stages
that were actually reached.

Each primary waypoint uses `moveToPositionAsync` with a finite timeout. After the
command completes, the script reads the measured state and checks the 3D position
error. If the vehicle is still outside tolerance, it attempts a small bounded
velocity-based correction using `moveByVelocityAsync`, then hovers, settles briefly,
and measures again. The waypoint is only accepted when the final measured position is
inside tolerance and collision-free.

## Commands

Run the default horizontal square:

```powershell
python scripts\run_colosseum_waypoint_demo.py
```

Run each individual route:

```powershell
python scripts\run_colosseum_waypoint_demo.py --route horizontal-square
python scripts\run_colosseum_waypoint_demo.py --route figure-eight
python scripts\run_colosseum_waypoint_demo.py --route vertical-square
```

Run the complete route suite:

```powershell
python scripts\run_colosseum_waypoint_demo.py --route all
```

Useful tuning options:

```powershell
python scripts\run_colosseum_waypoint_demo.py --route figure-eight --figure-eight-samples 13 --figure-eight-x-scale 3.0 --figure-eight-y-scale 2.0
python scripts\run_colosseum_waypoint_demo.py --route vertical-square --vertical-square-width 2.0 --vertical-square-height 1.0 --min-ground-clearance 1.0
python scripts\run_colosseum_waypoint_demo.py --waypoint-timeout 20 --waypoint-tolerance 0.5 --velocity 0.5
```

## Expected Output

During execution, progress output reports:

- route name
- waypoint number and total
- requested target position
- measured position
- 3D position error
- collision state

At the end of each route, the summary reports:

- waypoints requested and completed
- maximum position error
- final position error
- whether a collision occurred
- whether the route returned within tolerance of the measured anchor

## Troubleshooting

| Symptom | Suggested check |
|---|---|
| Waypoint tolerance exceeded | Increase tolerance slightly only if measured drift is minor, or reduce velocity/route size. |
| Ground-clearance validation fails | Lower vertical route height, increase anchor altitude, or reduce minimum ground clearance only if safe. |
| Missing `moveToPositionAsync` | Confirm the installed Python client matches Colosseum `v2.0.0-beta` at commit `7b9658a1`. |
| Missing `moveByVelocityAsync` | Waypoint correction requires this API only if the primary movement finishes outside tolerance. |
| Collision info unavailable | The script treats unavailable collision APIs as no collision but still validates position tracking. |
| Version or RPC errors | Recheck the M12.2 setup: editable install from `Colosseum/PythonClient` and `msgpack==0.6.2`. |
| `IOLoop is already running` after Ctrl+C | Restart Blocks and run a fresh Python process. The legacy msgpackrpc/Tornado client may be left in a bad state after an interrupted async join. |

## Manual Live Validation Checklist

1. Start Colosseum Blocks `v2.0.0-beta`.
2. Confirm the Python client is installed from Colosseum tag `v2.0.0-beta`, commit
   `7b9658a1`.
3. Confirm `msgpack==0.6.2`.
4. Run the M12.2 read-only connection check.
5. Run `horizontal-square`.
6. Run `figure-eight`.
7. Run `vertical-square`.
8. Run `all`.
9. Confirm the output reports return-to-anchor success and cleanup lands, disarms,
   and disables API control.


To show the UAV's continuous flight trace:

1. Click the Blocks simulator window.
2. Press T to enable tracing.
3. Run the waypoint demo.
4. Press T again to disable tracing.

The trace style can be configured with:

client.simSetTraceLine([1.0, 0.0, 0.0, 1.0], 5.0)
