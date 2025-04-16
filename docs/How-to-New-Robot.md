# Setting Up a New Robot from URDF

> ⚠️ **Important:** All mesh files **must be in meters**.

## 1. Organizing Your Robot Files

To set up a new robot, ensure you have its URDF file and corresponding mesh files. We recommend placing each robot in its own subfolder under the `models` directory:

```
models/
└── your_robot_name/
    ├── robot.urdf
    └── meshes/
        ├── link0.obj
        ├── link1.obj
        └── ...
```

## 2. Environment Setup

Set up a Python environment using the provided `requirements.txt`:

```bash
# From the project root directory:

# Create a new conda environment
conda create -n rtcd_env

# Activate the environment
conda activate rtcd_env

# Install dependencies
pip install -r requirements.txt
```

> If you're re-entering the environment later:
```bash
conda activate rtcd_env
```

## 3. Parsing the URDF

Run the URDF parser to convert your URDF into a format compatible with the Ray-Traced Collision Detector (RTCD):

```bash
python ./scripts/URDFParser/urdf_parser.py --urdf_file=./models/your_robot_name/robot.urdf
```

Optional flags:
- `--gen_sphere`: Automatically generate a sphere-based representation of the robot.

## 4. Output

The parsed data will be saved in `RTCD/robot/models/` as a `.h` header file. This file can later be compiled directly into the ray tracer.

---

## URDF Parser Details

Our custom URDF parser performs the following tasks:

- **Mesh normalization**: Applies pre-transformations to zero out scaling and offsets.
- **Link merging**: Merges fixed-connected links into a single mesh.
- **Graph rewiring**: Handles branching robots with non-linear joint hierarchies.
- **Kinematics setup**: Precomputes necessary data for forward kinematics using Rodrigues' rotation formula.
- **Sphere generation**: Computes an approximate sphere-based representation (if not already provided).

> 🔧 **Coming Soon**: A future release will export a single `.json` file and load robot data at runtime. It will also support on-demand mesh simplification.

---

## Sphere Representation

We use a KD-tree–based algorithm to generate spheres from triangle meshes:

1. Subdivide mesh edges to generate a near-uniform set of points.
2. Group points using a KD-tree structure.
3. Generate a sphere for each KD-tree node.

> 🔍 **Alternative**: [FOAM](https://github.com/CoMMALab/foam), developed by CoMMALab, uses the medial axis to generate high-quality sphere representations. However, its default settings may not work well on certain meshes due to memory issues.

---

## Known Limitations

- Supports **revolute joints** only (prismatic joints coming soon).
- **Floating-base** robots are not yet supported.
- May not handle unconventional joint structures.
- Sphere representation is currently **coarse** and best suited for prototyping.