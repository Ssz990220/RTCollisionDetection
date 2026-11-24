# configs/

JSON configuration: checker configs (mode, streams, capacities, build preference, culling, OptiX
compile options, ...), scene descriptions (obstacle list: mesh path, pose, scale) and the `robot` block that says
where the robot description (URDF) and its mesh / sphere geometry are and in which format. The JSON is parsed by
the `rtcd_json` module (`include/rtcd/json_config.h`, `src/json`) into the plain structs of `include/rtcd/`;
`rtcd_config` (`src/config`) validates and resolves the checker config and has no JSON dependency. Unspecified
fields get the C++ defaults, unknown keys are errors, and every relative path in a file is resolved against the
directory of that file, so configs work from any working directory and no source path is baked into a binary.
Schemas: `docs/design/interfaces.md` section 7. The regression inputs, goldens and stored references are committed
under `tests/data/` (see its README); the local, untracked `data/` folder only receives benchmark results.

| file | content |
|------|---------|
| `checker_ias.json`, `checker_ray.json`, `checker_linear.json`, `checker_quad.json` | the checker of each regression mode, as the goldens were recorded (old stack's `ccRegression.cpp`: 4 streams, batch 128, 4096 items, OBB on for IAS / RAY, off for curves, 16 trajectory points, 8 control points for QUAD) |
| `scenes/shelf.json` | "medium shelf", used by the goldens (old stack's `CollisionScenes/Scene/shelf.h`) |
| `scenes/shelf_dense.json` | dense shelf (old `Scene/denseShelf.h`) |
| `scenes/shelf_simple.json` | simple shelf (old `Scene/shelfSimple.h`) |
| `robots/panda.json` | Franka Panda: `models/franka_description/panda.urdf`, watertight simplified meshes, sphere files |
| `setups/regression_{ias,ray,linear,quad}.json` | checker + `scenes/shelf.json` + `robots/panda.json` for each regression mode (`rtcd::loadSetup`) |
| `setups/demo_{ias,ray,linear,quad}.json` | the demos' setups (`rtcd_demo_discrete` / `rtcd_demo_curve --setup demo_<mode>`): the old demo parameters on `scenes/shelf_dense.json` |
| `spline_weights/` | B-spline fitting weights `<quad|cubic>/<normal|repeated>/<control>_<trajectory>.txt`, an identical copy of the old stack's `RTCD/data/weights/` (manifest: `tests/data/reference/spline_weights.txt`; `curve.spline_weights_dir`) |
