# configs/

JSON configuration of the new stack and its golden references: checker configs (mode, streams, capacities, build
preference, culling, OptiX compile options, ...), scene descriptions (obstacle list: mesh path, pose, scale) and
the `robot` block that says where the robot description (URDF) and its mesh / sphere geometry are and in which
format. The JSON is parsed by the `rtcd_config` module into the plain structs of `include/rtcd/`; unspecified
fields get defaults, and every path in a file is resolved relative to that file, so configs work from any working
directory and no source path is baked into a binary. Large generated data (poses, trajectories, caches) stays in
the shared, untracked `data/` folder; the old stack's regression goldens remain in `data/CCTest/`.
