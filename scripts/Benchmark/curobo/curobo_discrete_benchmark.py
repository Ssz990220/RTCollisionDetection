import argparse
import time

# Third Party
import numpy as np
import torch
import pandas as pd

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.rollout.arm_base import ArmBase, ArmBaseConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import (
    get_robot_configs_path,
    get_robot_list,
    get_task_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
    write_yaml,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

motion_dir = "./data/motions/"
result_dir = "./data/Benchmark/result/discrete/"
nSrcPoses = 8192

# read a bin file and return the trajectory
def load_traj(fileName):
    traj = np.fromfile(fileName, dtype=np.float32)
    traj = traj.reshape(-1, 7)
    return traj


def load_curobo(robot_file, world_file):
    # load curobo arm base?

    world_cfg = load_yaml(join_path(get_world_configs_path(), world_file))

    base_config_data = load_yaml(join_path(get_task_configs_path(), "base_cfg.yml"))
    graph_config_data = load_yaml(join_path(get_task_configs_path(), "graph.yml"))
    # base_config_data["constraint"]["self_collision_cfg"]["weight"] = 0.0
    # if not compute_distance:
    #    base_config_data["constraint"]["primitive_collision_cfg"]["classify"] = False
    robot_config_data = load_yaml(join_path(get_robot_configs_path(), robot_file))

    arm_base = ArmBaseConfig.from_dict(
        robot_config_data["robot_cfg"],
        graph_config_data["model"],
        base_config_data["cost"],
        base_config_data["constraint"],
        base_config_data["convergence"],
        base_config_data["world_collision_checker_cfg"],
        world_cfg,
    )
    arm_base = ArmBase(arm_base)
    return arm_base


def bench_collision_curobo(robot_file, world_file, src_poses, b_size, n, use_cuda_graph=True):
    arm_base = load_curobo(robot_file, world_file)
    arm_base.robot_self_collision_constraint.disable_cost()     # disable self collision cost
    arm_base.bound_constraint.disable_cost()                    # disable bound cost
    # load graph module:
    # randomly sample a batch of test configs from src_poses of b_size
    q_test = torch.tensor(src_poses[np.random.choice(src_poses.shape[0], b_size, replace=False)], dtype=torch.float32)
    tensor_args = TensorDeviceType()
    q_test = tensor_args.to_device(q_test).unsqueeze(1)

    tensor_args = TensorDeviceType()
    q_warm = q_test + 0.5

    if not use_cuda_graph:
        q = q_warm.clone()
        
        ts = []
        for _ in range(n):
            q_test = torch.tensor(src_poses[np.random.choice(src_poses.shape[0], b_size, replace=False)], dtype=torch.float32)
            tensor_args = TensorDeviceType()
            q_test = tensor_args.to_device(q_test).unsqueeze(1)
            q.copy_(q_test.detach().requires_grad_(False))
            torch.cuda.synchronize()

            st_time = time.time()
            out = arm_base.rollout_constraint(q)
            torch.cuda.synchronize()
            dt = time.time() - st_time
            ts.append(dt)

        # return the median time
        dt = np.median(ts)

    else:
        q = q_warm.clone()

        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(3):
                out = arm_base.rollout_constraint(q_warm)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            out = arm_base.rollout_constraint(q)

        ts = []
        for _ in range(n):
            q_test = torch.tensor(src_poses[np.random.choice(src_poses.shape[0], b_size, replace=False)], dtype=torch.float32)
            tensor_args = TensorDeviceType()
            q_test = tensor_args.to_device(q_test).unsqueeze(1)
            q.copy_(q_test.detach().requires_grad_(False))
            st_time = time.time()
            g.replay()
            # a = out.feasible
            torch.cuda.synchronize()
            dt = time.time() - st_time
            ts.append(dt)

        dt = np.mean(ts)
        print("Collision Checking Time for {} samples: {} us, runtime variation: {} us".format(b_size, np.mean(ts) * 1e6, np.std(ts) * 1e6))
    return dt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    b_list = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    n_list = [125, 125, 125, 125, 125, 250, 250, 500, 500, 500, 1500, 2000, 2000]

    robot_list = get_robot_list()
    robot_list = [robot_list[0]]

    world_files = ["benchmark_shelf","benchmark_shelf_dense","benchmark_shelf_simple"]

    print("running...")
    for world in world_files:
        data = {"robot": [], "Collision Checking": [], "Batch Size": []}
        world_file = world + ".yml"
        for robot_file in robot_list:
            arm_sampler = load_curobo(robot_file, world_file)

            counter = 1
            src_poses = load_traj(motion_dir + "Panda" + str(nSrcPoses) + ".bin")
            # create a sampler with dof:
            for n, b_size in zip(n_list, b_list):

                dt_cu_cg = bench_collision_curobo(
                    robot_file,
                    world_file,
                    src_poses,
                    b_size,
                    n,
                    use_cuda_graph=True,
                )
                
                data["robot"].append(robot_file)
                data["Collision Checking"].append(float(dt_cu_cg))
                data["Batch Size"].append(b_size)
        write_yaml(data, result_dir + "kinematics" + "_" + world + ".yml")
