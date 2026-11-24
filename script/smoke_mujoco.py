"""Minimal MuJoCo smoke test for TableSim."""

from __future__ import annotations

import mujoco
import numpy as np


XML = """
<mujoco model="tablesim_smoke">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="box" pos="0 0 0.15">
      <joint type="free"/>
      <geom type="box" size="0.03 0.03 0.03" rgba="0.2 0.6 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main() -> int:
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    for _ in range(10):
        mujoco.mj_step(model, data)
    try:
        renderer = mujoco.Renderer(model, width=320, height=240)
        renderer.update_scene(data)
        frame = renderer.render()
        assert frame.shape[:2] == (240, 320)
        del renderer
    except Exception as exc:
        raise SystemExit(f"MuJoCo render smoke test failed: {exc}") from exc
    print("TableSim MuJoCo smoke test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

