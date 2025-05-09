# cuda_test.py
import numpy as np
import time

from numba import cuda
from organisms_cuda import _move_kernel


def run_test(N=10000):
    # Initialize random data for N organisms
    rng = np.random.RandomState(0)
    x_pos       = rng.uniform(0, 100, size=N).astype(np.float32)
    y_pos       = rng.uniform(0, 100, size=N).astype(np.float32)
    energy      = rng.uniform(0, 100, size=N).astype(np.float32)
    camouflage  = rng.uniform(0, 1,   size=N).astype(np.float32)
    vision      = rng.uniform(0, 10,  size=N).astype(np.float32)
    attack      = rng.uniform(0, 1,   size=N).astype(np.float32)
    defense     = rng.uniform(0, 1,   size=N).astype(np.float32)
    pack_flag   = np.zeros(N, dtype=np.bool_)
    swim        = np.zeros(N, dtype=np.bool_)
    walk        = np.ones(N,  dtype=np.bool_)
    fly         = np.zeros(N, dtype=np.bool_)
    speed       = rng.uniform(0.1, 1, size=N).astype(np.float32)
    species     = rng.randint(0, 5,  size=N).astype(np.int32)

    # Dummy environment grid and direction offsets
    terrain_h, terrain_w = 200, 200
    terrain     = np.zeros((terrain_h, terrain_w), dtype=np.float32)
    dirs        = np.array([[1,0],[-1,0],[0,1],[0,-1]], dtype=np.float32)
    dir_count   = dirs.shape[0]
    # No neighbors for simplicity: each organism has an empty neighbor list
    neigh_ptrs  = np.arange(N+1, dtype=np.int32)
    neigh_inds  = np.empty(0,   dtype=np.int32)

    # Copy data to GPU
    d_x       = cuda.to_device(x_pos)
    d_y       = cuda.to_device(y_pos)
    d_e       = cuda.to_device(energy)
    d_cam     = cuda.to_device(camouflage)
    d_vis     = cuda.to_device(vision)
    d_att     = cuda.to_device(attack)
    d_def     = cuda.to_device(defense)
    d_pack    = cuda.to_device(pack_flag)
    d_swim    = cuda.to_device(swim)
    d_walk    = cuda.to_device(walk)
    d_fly     = cuda.to_device(fly)
    d_speed   = cuda.to_device(speed)
    d_spc     = cuda.to_device(species)
    d_terrain = cuda.to_device(terrain)
    d_dirs    = cuda.to_device(dirs)
    d_dirs_count = cuda.to_device(dir_count)
    d_ptrs    = cuda.to_device(neigh_ptrs)
    d_inds    = cuda.to_device(neigh_inds)

    # Allocate output arrays on GPU
    d_new_x = cuda.device_array_like(d_x)
    d_new_y = cuda.device_array_like(d_y)
    d_new_e = cuda.device_array_like(d_e)

    # Configure launch
    threads_per_block = 128
    blocks_per_grid  = (N + threads_per_block - 1) // threads_per_block

    # Warm-up kernel launch
    _move_kernel[blocks_per_grid, threads_per_block](
        d_x, d_y, d_e,
        d_cam, d_vis, d_att, d_def,
        d_pack, d_swim, d_walk, d_fly, d_speed,
        d_spc,
        d_terrain, terrain_w, terrain_h,
        d_dirs, d_dirs_count, d_ptrs, d_inds,
        d_new_x, d_new_y, d_new_e
    )
    cuda.synchronize()

    # Timed run
    start = time.perf_counter()
    _move_kernel[blocks_per_grid, threads_per_block](
        d_x, d_y, d_e,
        d_cam, d_vis, d_att, d_def,
        d_pack, d_swim, d_walk, d_fly, d_speed,
        d_spc,
        d_terrain, terrain_w, terrain_h,
        d_dirs, d_ptrs, d_inds,
        d_new_x, d_new_y, d_new_e
    )
    cuda.synchronize()
    end = time.perf_counter()

    print(f"Processed {N} organisms on GPU in {end - start:.6f} seconds")


if __name__ == "__main__":
    run_test()
