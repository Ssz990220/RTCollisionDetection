import argparse
import time

# Third Party
import numpy as np
import pandas as pd
import torch
import os

# set numpy random seed
np.random.seed(0)

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
    get_module_path
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

path = os.path.dirname(__file__)
dataDir = os.path.join(path,"../../data/motions/")
os.makedirs(dataDir, exist_ok=True)
result_dir = os.path.join(path,"../../data/Benchmark/result/curve_loop/")
os.makedirs(result_dir, exist_ok=True)
weightStr = "1111111"
KNN_K = 4

def load_traj(nCSpaceSmpl):
    # load bin file contains float from ../../data/traj.bin
    traj = np.fromfile(join_path(dataDir, "Panda" + str(nCSpaceSmpl) + ".bin"), dtype=np.float32)
    traj = traj.reshape(-1, 7)
    return traj

def load_knn_idx(nCSpaceSmpl, knn_k):
    # load bin file contains int from ../../data/knn.bin
    knn_idx = np.fromfile(join_path(dataDir, "Panda/{}/{}_knn{}_idx.bin".format(weightStr, nCSpaceSmpl, knn_k)), dtype=np.int32)
    return knn_idx.reshape(-1, knn_k)

def interpolate_traj(poses, knn_idx, h):
    # create a 3D tensor from poses of dimension (nTraj, h, 7)
    # in the ith trajectory, the first pose is poses[i], the second pose is poses[knn_idx[i]]
    # the poses in between follows linear interpolation pattern
    nTraj = poses.shape[0] * knn_idx.shape[1]
    q = np.zeros((nTraj, h, 7), dtype=np.float32)
    counter = 0
    for i in range(poses.shape[0]):
        q[counter, 0, :] = poses[i,:]
        for j in range(knn_idx.shape[1]):
            q[counter, :, :] = np.linspace(poses[i, :], poses[knn_idx[i, j], :], h, endpoint=True)
            counter += 1

    return q

def load_curobo(robot_file, world_file, horizon):

    world_cfg = load_yaml(join_path(get_world_configs_path(), world_file))

    base_config_data = load_yaml(join_path(get_task_configs_path(), "base_cfg_sweep.yml"))
    graph_config_data = load_yaml(join_path(get_task_configs_path(), "graph_sweep.yml"))
    graph_config_data["model"]["horizon"] = horizon
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


def bench_collision_curobo(robot_file, world_file, src_traj, b_size, n, use_cuda_graph=True, horizon = 1):
    arm_base = load_curobo(robot_file, world_file, horizon)
    arm_base.robot_self_collision_constraint.disable_cost()
    arm_base.bound_constraint.disable_cost()

    if b_size > src_traj.shape[0]:
        return 0

    if not use_cuda_graph:
        # randomly sample a batch of test trajs from src_traj of b_size
        q_test = torch.tensor(src_traj[np.random.choice(src_traj.shape[0], b_size, replace=False)], dtype=torch.float32)
        # load graph module:
        tensor_args = TensorDeviceType()
        q_test = tensor_args.to_device(q_test).unsqueeze(1)

        tensor_args = TensorDeviceType()
        q_warm = q_test + 0.5
        
        q = q_warm.clone()
        for _ in range(10):
            out = arm_base.rollout_constraint(q_warm)
            torch.cuda.synchronize()

        torch.cuda.synchronize()

        

        dt = []
        nTimes = n
        for _ in range(nTimes):
            q_test = torch.tensor(src_traj[np.random.choice(src_traj.shape[0], b_size, replace=False)], dtype=torch.float32)
            tensor_args = TensorDeviceType()
            q_test = tensor_args.to_device(q_test).unsqueeze(1)
            q.copy_(q_test.detach().requires_grad_(False))

            st_time = time.time()
            out = arm_base.rollout_constraint(q)
            torch.cuda.synchronize()
            dt.append(time.time() - st_time)
    else:
        
        q = torch.tensor(src_traj[np.random.choice(src_traj.shape[0], b_size, replace=False)], dtype=torch.float32)
        tensor_args = TensorDeviceType()
        q = tensor_args.to_device(q).unsqueeze(1)
        q_warm = q + 0.5

        dt = []
        nTimes = n
        s = torch.cuda.Stream()

        g = torch.cuda.CUDAGraph()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(3):
                out = arm_base.rollout_constraint(q_warm)

        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            out = arm_base.rollout_constraint(q)

        for _ in range(nTimes):
            
            choices = np.random.choice(src_traj.shape[0], b_size, replace=False)
            selected_traj = src_traj[choices]
            q_test = torch.tensor(selected_traj, dtype=torch.float32)
            tensor_args = TensorDeviceType()
            q_test = tensor_args.to_device(q_test).unsqueeze(1)

            q.copy_(q_test.detach().requires_grad_(False))
            
            for _ in range(5):
                g.replay()

            torch.cuda.synchronize()

            t0 = time.time()
            g.replay()  # or the no-graph version
            torch.cuda.synchronize()
            t1 = time.time()
            a = out.feasible
            # print(a)
            elapsed = (t1 - t0)
            dt.append(elapsed)

        torch.cuda.synchronize()

    print("Collision Checking Time for {} samples: {} us, runtime variation: {} us".format(b_size, np.mean(dt) * 1e6, np.std(dt) * 1e6))
    return np.mean(dt)



if __name__ == "__main__":

    robot_list = get_robot_list()
    robot_file = robot_list[0]
    trajLengths = [8, 16]
    nCSpaceSamples = [8192]
    b_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    n_list = [4000, 4000, 1500, 500, 500, 500, 250, 250, 125, 125, 125, 125, 125]
    # reverse the order of b_list and n_list
    b_list = b_list[::-1]
    n_list = n_list[::-1]

    world_files = ["benchmark_shelf","benchmark_shelf_dense", "benchmark_shelf_simple"]
    # world_files = ["benchmark_shelf_simple"]

    print("running...")

    for world in world_files:
        data = {"sweep_steps":[], "Collision Checking": [], "Batch Size": []}
        for nCSpaceSample in nCSpaceSamples:
            poses = load_traj(nCSpaceSample)
            knn_idx = load_knn_idx(nCSpaceSample, KNN_K)
            print("\nBenchmarking {} with {} CSpace Samples".format(world, nCSpaceSample))
            for h in trajLengths:
                print("Benchmarking {} with {} CSpace Samples and {} sweep_steps".format(world, nCSpaceSample, h))
                qs = interpolate_traj(poses, knn_idx, h)
                print("CUDA Graph On")
                for n, b_size in zip(n_list, b_list):
                    dt_cu_cg = bench_collision_curobo(
                        robot_file,
                        world + ".yml",
                        qs,
                        b_size,
                        n,
                        use_cuda_graph=True,
                        horizon=h
                    )
                    data["sweep_steps"].append(h)
                    data["Collision Checking"].append(float(dt_cu_cg))
                    data["Batch Size"].append(b_size)
        write_yaml(data, result_dir + "curobo_swept_{}.yml".format(world))
        df = pd.DataFrame(data)
