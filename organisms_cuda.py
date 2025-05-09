# organisms_cuda_fixed.py
import numpy as np
from numba import cuda

@cuda.jit

def _move_kernel(
    x_pos, y_pos, energy_in,
    camouflage, vision, attack, defense,
    pack_flag, swim, walk, fly, speed,
    species,
    terrain, terrain_width, terrain_height,
    dirs, dir_count, neigh_ptrs, neigh_inds,
    new_x, new_y, energy_out
):
    i = cuda.grid(1)
    N = x_pos.shape[0]
    if i >= N:
        return

    # 1) Energy penalty with bounds check
    xi = int(x_pos[i]); yi = int(y_pos[i])
    if 0 <= xi < terrain_width and 0 <= yi < terrain_height:
        land = terrain[yi, xi] >= 0
    else:
        land = False
    swim_only = swim[i] & (~walk[i]) & (~fly[i])
    walk_only = walk[i] & (~swim[i]) & (~fly[i])
    e = energy_in[i]
    if (swim_only and land) or (walk_only and not land):
        e -= 5.0

    # 2) Terrain avoidance
    aw_x = 0.0; aw_y = 0.0
    al_x = 0.0; al_y = 0.0
    for d in range(dir_count):  # dir_count is a native Python int
        dx = dirs[2 * d]; dy = dirs[2 * d + 1]
        nx = xi + int(dx); ny = yi + int(dy)
        if 0 <= nx < terrain_width and 0 <= ny < terrain_height:
            t = terrain[ny, nx]
            if t < 0:
                aw_x -= dx; aw_y -= dy
            else:
                al_x -= dx; al_y -= dy

    # 3) Neighbors
    start = neigh_ptrs[i]; end = neigh_ptrs[i + 1]
    if start >= neigh_inds.size:
        new_x[i] = x_pos[i]
        new_y[i] = y_pos[i]
        energy_out[i] = e
        return

    # 4) Steering accumulators
    my_cam = camouflage[i]
    my_att = attack[i]; my_def = defense[i]
    my_spd = speed[i]; my_spc = species[i]

    steer_x = 0.0; steer_y = 0.0
    cnt_h = 0; cnt_p = 0; cnt_s = 0; cnt_c = 0
    sum_hx = 0.0; sum_hy = 0.0
    sum_px = 0.0; sum_py = 0.0
    sum_sx = 0.0; sum_sy = 0.0
    sum_cx = 0.0; sum_cy = 0.0

    if pack_flag[i]:
        # pack logic
        for idx in range(start, end):
            j = neigh_inds[idx]
            if j < 0 or j >= N or j == i:
                continue
            if vision[j] < my_cam:
                continue
            dx = x_pos[j] - x_pos[i]; dy = y_pos[j] - y_pos[i]
            their_net = attack[j] - my_def
            my_net    = my_att - defense[j]
            if not pack_flag[j]:
                if their_net > my_net:
                    sum_hx += x_pos[j]; sum_hy += y_pos[j]; cnt_h += 1
                elif my_net > their_net:
                    sum_px += x_pos[j]; sum_py += y_pos[j]; cnt_p += 1
            else:
                sum_sx += x_pos[j]; sum_sy += y_pos[j]; cnt_s += 1
                if dx*dx + dy*dy < 25.0:
                    sum_cx += dx; sum_cy += dy; cnt_c += 1
        # compose steer
        if cnt_h > 0:
            steer_x = x_pos[i] - (sum_hx / cnt_h)
            steer_y = y_pos[i] - (sum_hy / cnt_h)
        elif cnt_p > 0:
            steer_x = (sum_px / cnt_p) - x_pos[i]
            steer_y = (sum_py / cnt_p) - y_pos[i]
        elif cnt_s > 0:
            steer_x = (sum_sx / cnt_s) - x_pos[i]
            steer_y = (sum_sy / cnt_s) - y_pos[i]
            if cnt_c > 0:
                steer_x += (-sum_cx / cnt_c) * 10.0
                steer_y += (-sum_cy / cnt_c) * 10.0
    else:
        # social steering
        for idx in range(start, end):
            j = neigh_inds[idx]
            if j < 0 or j >= N or j == i:
                continue
            if vision[j] < my_cam:
                continue
            dx = x_pos[j] - x_pos[i]; dy = y_pos[j] - y_pos[i]
            their_net = attack[j] - my_def
            my_net    = my_att - defense[j]
            if their_net > my_net:
                sum_hx += x_pos[i] - x_pos[j]; sum_hy += y_pos[i] - y_pos[j]; cnt_h += 1
            elif my_net > their_net:
                sum_px += x_pos[j] - x_pos[i]; sum_py += y_pos[j] - y_pos[i]; cnt_p += 1
            if species[j] == my_spc:
                sum_cx += x_pos[i] - x_pos[j]; sum_cy += y_pos[i] - y_pos[j]; cnt_c += 1
        # compose steer
        if cnt_h > 0:
            steer_x += sum_hx / cnt_h; steer_y += sum_hy / cnt_h
        if cnt_p > 0:
            steer_x += sum_px / cnt_p; steer_y += sum_py / cnt_p
        if cnt_c > 0:
            steer_x += (sum_cx / cnt_c) * (0.5 * my_spd)

    # 5) Add terrain avoidance
    if not swim[i]:
        steer_x += 5.0 * aw_x; steer_y += 5.0 * aw_y
    if not walk[i]:
        steer_x += 5.0 * al_x; steer_y += 5.0 * al_y

    # 6) Normalize & step
    mag = (steer_x*steer_x + steer_y*steer_y) ** 0.5
    if mag > 0.0:
        step_x = (steer_x / mag) * my_spd
        step_y = (steer_y / mag) * my_spd
    else:
        step_x = 0.0; step_y = 0.0

    # 7) Write and clip
    nx = x_pos[i] + step_x; ny = y_pos[i] + step_y
    if nx < 0: nx = 0
    elif nx >= terrain_width: nx = terrain_width - 1
    if ny < 0: ny = 0
    elif ny >= terrain_height: ny = terrain_height - 1

    new_x[i] = nx
    new_y[i] = ny
    energy_out[i] = e
