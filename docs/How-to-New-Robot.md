# Setting up a new robot

The robot is loaded **at run time** from its URDF; nothing is generated or compiled per robot (the old stack's
generated `RTCD/Robot/models/<robot>.h` header is gone together with the old stack). A robot needs three things,
referenced from a robot file such as `configs/robots/panda.json`:

1. a URDF (kinematics, joint limits, which mesh belongs to which link),
2. one triangle mesh per link (OBJ, meters, link frame),
3. for the curve modes only: a sphere approximation per link (`xyzr` text files).

The key `"robot"` of a setup file points at the robot file (setup files: [configuration reference](user/configuration.md#setup-file)).
This document is the authoritative description of the rules; the JSON keys are listed in the
[configuration reference](user/configuration.md#robot-file).

> **All meshes must be in meters.**

## 1. Organizing the robot files

Put each robot in its own folder under `models/`:
```
models/
└── your_robot/
    ├── your_robot.urdf
    ├── meshes/          link0.obj, link1.obj, ...
    └── spheres/         <link name>.txt (curve modes)
```

## 2. The robot file

```json
{
  "robot": {
    "description": "../../models/franka_description/panda.urdf",
    "geometry": {
      "mesh":    { "dir": "../../models/franka_description/meshes/Simplified/watertight", "format": "obj", "source": "visual" },
      "spheres": { "dir": "../../models/franka_description/spheres", "format": "xyzr" }
    }
  }
}
```
Paths are relative to the robot file. Optional keys: `"geometry.mesh.files"` / `"geometry.spheres.files"`
(`{ "<link name>": "<file>" }`, per-link overrides, relative to that `dir`), `"base_pose"` (robot root in the world,
a pose object as in the scene file; default identity). `spheres` is required by the curve modes only.

## 3. URDF rules (kinematics, joint limits, which mesh per link)

Read with urdfdom when the robot is loaded:
- Joints: `revolute` and `continuous` are actuated (continuous joints get the limits [-pi, pi]; a revolute joint
  must have a `<limit>`); `fixed` joints are folded into the transforms; `prismatic`, `floating`, `planar` and
  mimic joints are rejected (`UnsupportedError` naming the joint).
- Links attached by a fixed joint are merged into their parent link (geometry transformed into the parent frame
  and appended); fixed-attached links without geometry disappear. A chain of geometry-less links joined by fixed
  joints above the first link with geometry (e.g. a `world` link) is folded into the base pose.
- After merging the robot must be a **serial chain**: link 0 is the root, link j + 1 the child of actuated joint j.
  A branch with two actuated joints is rejected. **1 <= DOF <= 16** (`kMaxDof` in `include/rtcd/limits.h`).
- Joint values in a pose are in chain order root to tip (`RobotModel::jointNames()`), radians.

## 4. Link meshes (`geometry.mesh`)

Every link of the chain, link 0 included, must end up with a mesh:
- The URDF element of each link selected by `"source"` (`"visual"`, default, or `"collision"`) must be one
  `<mesh>`: several `<visual>` (resp. `<collision>`) elements on one link and primitive shapes (box, cylinder,
  sphere) are rejected. Its `<origin>` and its `scale` are applied; only uniform positive scales are accepted.
- File: `mesh.files[<link name>]` if given, otherwise `mesh.dir / <file name of the URDF filename>`: the directory
  part of the URDF path is dropped (`meshes/Simplified/watertight/link7_nohand.obj` -> `<dir>/link7_nohand.obj`,
  also for `package://...` URIs).
- Format `"obj"` only (tinyobjloader, triangulated); units meters, link frame. The loader does not require
  watertight robot meshes; the Panda setup uses simplified watertight meshes. Mesh size matters for speed:
  `mesh` mode builds one GAS per link mesh, `ray` mode traces every unique edge of every link mesh.

## 5. Sphere files (`geometry.spheres`, curve modes)

The robot's approximation by spheres, one file per URDF link that has geometry, including links merged by fixed
joints:
- File: `spheres.files[<link name>]` if given, otherwise `spheres.dir / <link name>.txt`
  (`models/franka_description/spheres/panda_link0.txt` ... `panda_link7.txt`). A missing file is an `IoError`.
- Format `"xyzr"`: UTF-8 text, one sphere per line, `x y z r` (four floats, whitespace separated, link frame,
  meters, `r > 0`); empty lines and lines starting with `#` are skipped; any other line is an `IoError` with its
  line number. An empty file gives the link zero spheres. Spheres of merged links are transformed into the parent
  link frame.

### Generating spheres with the URDF parser tool (`scripts/URDFParser`)

`scripts/URDFParser/urdf_parser.py` merges fixed-connected links, writes a modified URDF and, with `--gen_sphere`,
a sphere file per link. Python environment (from the repository root):
```bash
conda create -n rtcd_env
conda activate rtcd_env
pip install -r requirements.txt
python ./scripts/URDFParser/urdf_parser.py --urdf_file=./models/your_robot/your_robot.urdf --gen_sphere
```
What it writes, next to the URDF:
- `modified_<robot>.urdf`: the robot with every fixed-connected group merged into one link (the link closest to
  the root keeps its name), visual / collision meshes pointing at `merged_meshes/<link>_visual.obj` /
  `_collision.obj` with the URDF origins and scales already applied (`modified_panda.urdf` is the Panda result);
- `merged_meshes/`: those merged meshes;
- `spheres/<link>.txt` (with `--gen_sphere`): one line `x y z r` per sphere, generated from the merged visual mesh
  of each link (edges subdivided to <= 5 cm, a KD-tree of depth 3 over the vertices, one bounding sphere per leaf);
- last, a C++ header for the removed old stack (`./RTCD/Robot/models/<robot>.h`, relative to the working
  directory) and the masks printed to the console: **not used** any more (the checker derives the kinematics from
  the URDF at run time). Since `RTCD/` no longer exists and the script does not create the folder, this last step
  presumably ends with a `FileNotFoundError` (read from the code, not run); everything above is written before it.
  Create an empty `RTCD/Robot/models/` folder if you want the script to finish cleanly.

Compatibility with the loader (checked by reading the scripts, `sphereGenUtils.py` / `urdf_parser.py`
`genLinkSphere`; the Python tools themselves were not run): the file format is the `xyzr` format above (four
whitespace-separated numbers per line, Python's float formatting, which `std::istream >> float` reads) and the
naming rule is `<link name>.txt`, so `spheres.dir` can point at the generated `spheres/` folder. The spheres are in
the frame of the **merged** link. Therefore load the robot from `modified_<robot>.urdf` with
`mesh.dir = merged_meshes`: its links are exactly the files the tool wrote. With the original URDF instead, a link
that the tool merged into its parent would be merged again by the loader, which then also expects
`<child link>.txt` (not written: `IoError`) and would add the child's spheres a second time. The tool supports
revolute joints only and its spheres are coarse (good for prototyping); [FOAM](https://github.com/CoMMALab/foam)
(medial-axis based) is an alternative generator. Any tool that writes the `xyzr` format works.

## 6. Checking a new robot

Without a GPU: `rtcd::RobotModel::load(rtcd::loadRobotDesc("my_robot.json"))` throws a precise error for each rule
above and reports DOF, joint names, limits and per-link geometry. Then reference the robot file from a setup file
and run `rtcd_example` on it. Only the Franka Panda has been tested; the features beyond it (fixed-joint merging,
URDF origins and scales, continuous joints, arbitrary joint axes) are implemented but not covered by any test robot.

## Known limitations

- Revolute and continuous joints only (no prismatic joints, no floating base).
- Serial chains only (no branching after merging fixed links).
- No self-collision checking.
