"""Minimal Isaac Sim smoke test for TableSim."""

from __future__ import annotations


def main() -> int:
    try:
        from isaacsim import SimulationApp  # type: ignore
    except Exception:
        from omni.isaac.kit import SimulationApp  # type: ignore

    simulation_app = SimulationApp({"headless": True})
    try:
        simulation_app.update()
        print("TableSim Isaac Sim smoke test OK")
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

