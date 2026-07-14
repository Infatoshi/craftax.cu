#include "craftax.cuh"
#include <cmath>

// ============================================================
// Perlin Noise for World Generation
// ============================================================
__device__ float perlin_interp(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

__device__ float perlin_2d(float x, float y, const float* angles, int grid_size) {
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    float fx = x - x0, fy = y - y0;
    float u = perlin_interp(fx), v = perlin_interp(fy);

    auto grad_dot = [&](int ix, int iy, float dx, float dy) -> float {
        int idx = ((ix % grid_size) + grid_size) % grid_size * grid_size +
                  ((iy % grid_size) + grid_size) % grid_size;
        float a = angles[idx];
        return __cosf(a) * dx + __sinf(a) * dy;
    };

    float n00 = grad_dot(x0, y0, fx, fy);
    float n10 = grad_dot(x0+1, y0, fx-1, fy);
    float n01 = grad_dot(x0, y0+1, fx, fy-1);
    float n11 = grad_dot(x0+1, y0+1, fx-1, fy-1);

    float nx0 = n00 + u * (n10 - n00);
    float nx1 = n01 + u * (n11 - n01);
    return (nx0 + v * (nx1 - nx0) + 1.0f) * 0.5f;
}

// ============================================================
// World Generation (Reset)
//
// All RNG draws live at FIXED Philox counter offsets in the
// (seed, subsequence) stream, so tiles can be generated in any
// order -- serially by one thread or cooperatively by a warp --
// with bit-identical results. Draw budget per world:
//   [0, 400)                  Perlin angle table (4 layers x 100)
//   [WG_TILE_OFF + 2t, +2t+2) tile t: ore roll, tree roll
//   [WG_DIAMOND_OFF + 2a, ..) diamond fallback attempt a: r, c
//   [WG_GAMEPLAY_OFF, ...)    gameplay draws after reset
// ============================================================
#define WG_GRID 10
#define WG_ANGLE_COUNT (4 * WG_GRID * WG_GRID)
#define WG_TILE_OFF 512
#define WG_DIAMOND_OFF (WG_TILE_OFF + 2 * MAP_SIZE * MAP_SIZE)
#define WG_GAMEPLAY_OFF 32768

__device__ __forceinline__ float philox_f_at(uint64_t seed, uint64_t subseq, uint64_t offset) {
    curandStatePhilox4_32_10_t st;
    curand_init(seed, subseq, offset, &st);
    return curand_uniform(&st);
}

__device__ __forceinline__ uint32_t philox_u32_at(uint64_t seed, uint64_t subseq, uint64_t offset) {
    curandStatePhilox4_32_10_t st;
    curand_init(seed, subseq, offset, &st);
    return curand(&st);
}

// Classify one tile given the angle table. Pure function of
// (seed, subseq, r, c, angles) -- no draw-order dependence.
__device__ int8_t wg_classify_tile(
    uint64_t seed, uint64_t subseq, const float* angles, int r, int c
) {
    const float scale = (float)MAP_SIZE / (float)(WG_GRID - 1);
    const int center = MAP_SIZE / 2;
    float nr = (float)r / scale;
    float nc = (float)c / scale;

    float water_noise = perlin_2d(nr, nc, angles + 0 * WG_GRID * WG_GRID, WG_GRID);
    float mountain_noise = perlin_2d(nr, nc, angles + 1 * WG_GRID * WG_GRID, WG_GRID);
    float tree_noise = perlin_2d(nr, nc, angles + 2 * WG_GRID * WG_GRID, WG_GRID);
    float path_noise = perlin_2d(nr, nc, angles + 3 * WG_GRID * WG_GRID, WG_GRID);

    float dist = sqrtf((float)((r-center)*(r-center) + (c-center)*(c-center)));
    float prox = 1.0f - min_f(dist / 20.0f, 1.0f);

    float water_val = water_noise - prox * 0.3f;
    float mountain_val = mountain_noise - prox * 0.3f;

    int8_t blk = BLK_GRASS;

    if (water_val > 0.7f) {
        blk = BLK_WATER;
    } else if (water_val > 0.6f && water_val <= 0.75f) {
        blk = BLK_SAND;
    } else if (mountain_val > 0.7f) {
        blk = BLK_STONE;
        if (path_noise > 0.8f) blk = BLK_PATH;
        if (mountain_val > 0.85f && water_noise > 0.4f) blk = BLK_PATH;
        if (mountain_val > 0.85f && tree_noise > 0.7f) blk = BLK_LAVA;
    }

    int tile = r * MAP_SIZE + c;
    if (blk == BLK_STONE) {
        float ore_roll = philox_f_at(seed, subseq, WG_TILE_OFF + 2 * (uint64_t)tile);
        if (ore_roll < 0.005f && mountain_val > 0.8f)
            blk = BLK_DIAMOND;
        else if (ore_roll < 0.035f)
            blk = BLK_IRON;
        else if (ore_roll < 0.075f)
            blk = BLK_COAL;
    }

    if (blk == BLK_GRASS && tree_noise > 0.5f
        && philox_f_at(seed, subseq, WG_TILE_OFF + 2 * (uint64_t)tile + 1) > 0.8f)
        blk = BLK_TREE;

    return blk;
}

// Reset all non-map per-env fields (everything except the map).
__device__ void wg_reset_fields(EnvSoA& g, int e, uint64_t seed, int rng_subseq) {
    const int n = g.n;
    const int center = MAP_SIZE / 2;
    g.player_r[e] = center; g.player_c[e] = center;
    g.player_dir[e] = 4;
    g.health[e] = 9; g.food[e] = 9; g.drink[e] = 9; g.energy[e] = 9;
    g.is_sleeping[e] = 0;
    g.recover[e] = 0; g.hunger[e] = 0; g.thirst[e] = 0; g.fatigue[e] = 0;

    for (int i = 0; i < NUM_INVENTORY; i++) g.inv[i*n+e] = 0;

    for (int i = 0; i < MAX_ZOMBIES; i++) { g.zombie_mask[i*n+e] = 0; g.zombie_hp[i*n+e] = 0; g.zombie_cd[i*n+e] = 0; }
    for (int i = 0; i < MAX_COWS; i++) { g.cow_mask[i*n+e] = 0; g.cow_hp[i*n+e] = 0; }
    for (int i = 0; i < MAX_SKELETONS; i++) { g.skel_mask[i*n+e] = 0; g.skel_hp[i*n+e] = 0; g.skel_cd[i*n+e] = 0; }
    for (int i = 0; i < MAX_ARROWS; i++) { g.arrow_mask[i*n+e] = 0; }
    for (int i = 0; i < MAX_PLANTS; i++) { g.plant_mask[i*n+e] = 0; g.plant_age[i*n+e] = 0; }

    g.ach[e] = 0;
    g.timestep[e] = 0;
    g.light_level[e] = 1.0f;

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, rng_subseq, WG_GAMEPLAY_OFF, &rng);
    g.rng[e] = rng;
}

// Serial worldgen (one thread per env). Bit-identical to the warp
// version below; used by the initial full reset and the torch ext.
__device__ void generate_world(EnvSoA& g, int e, uint64_t seed, int rng_subseq) {
    uint8_t* __restrict__ map = env_map(g, e);

    float angles[WG_ANGLE_COUNT];
    for (int i = 0; i < WG_ANGLE_COUNT; i++)
        angles[i] = philox_f_at(seed, rng_subseq, i) * 2.0f * 3.14159265f;

    const int center = MAP_SIZE / 2;
    bool has_diamond = false;
    for (int r = 0; r < MAP_SIZE; r++) {
        for (int c2 = 0; c2 < MAP_SIZE; c2 += 2) {
            int8_t b0 = (r == center && c2 == center) ? BLK_GRASS
                : wg_classify_tile(seed, rng_subseq, angles, r, c2);
            int8_t b1 = (r == center && c2 + 1 == center) ? BLK_GRASS
                : wg_classify_tile(seed, rng_subseq, angles, r, c2 + 1);
            has_diamond |= (b0 == BLK_DIAMOND) || (b1 == BLK_DIAMOND);
            map[r * MAP_PACKED_ROW + (c2 >> 1)] = (uint8_t)((b0 & 0x0F) | ((b1 & 0x0F) << 4));
        }
    }

    if (!has_diamond) {
        for (int attempts = 0; attempts < 1000; attempts++) {
            int r = (int)(philox_u32_at(seed, rng_subseq, WG_DIAMOND_OFF + 2 * (uint64_t)attempts) % MAP_SIZE);
            int c = (int)(philox_u32_at(seed, rng_subseq, WG_DIAMOND_OFF + 2 * (uint64_t)attempts + 1) % MAP_SIZE);
            if (map_get(map, r, c) == BLK_STONE) {
                map_set(map, r, c, BLK_DIAMOND);
                break;
            }
        }
    }

    wg_reset_fields(g, e, seed, rng_subseq);
}

// Warp-cooperative worldgen: 32 lanes generate one map together.
// Each lane owns byte pairs (2 adjacent tiles) so nibble packing
// never races. angles_smem must hold WG_ANGLE_COUNT floats of
// warp-private shared memory. Bit-identical to generate_world.
__device__ void generate_world_warp(
    EnvSoA& g, int e, uint64_t seed, int rng_subseq, float* angles_smem
) {
    const int lane = threadIdx.x & 31;
    uint8_t* __restrict__ map = env_map(g, e);

    for (int i = lane; i < WG_ANGLE_COUNT; i += 32)
        angles_smem[i] = philox_f_at(seed, rng_subseq, i) * 2.0f * 3.14159265f;
    __syncwarp();

    const int center = MAP_SIZE / 2;
    bool has_diamond = false;
    // 2048 packed bytes, lane-strided: lane handles bytes lane, lane+32, ...
    for (int byte = lane; byte < MAP_PACKED_SIZE; byte += 32) {
        int r = byte / MAP_PACKED_ROW;
        int c2 = (byte % MAP_PACKED_ROW) * 2;
        int8_t b0 = (r == center && c2 == center) ? BLK_GRASS
            : wg_classify_tile(seed, rng_subseq, angles_smem, r, c2);
        int8_t b1 = (r == center && c2 + 1 == center) ? BLK_GRASS
            : wg_classify_tile(seed, rng_subseq, angles_smem, r, c2 + 1);
        has_diamond |= (b0 == BLK_DIAMOND) || (b1 == BLK_DIAMOND);
        map[byte] = (uint8_t)((b0 & 0x0F) | ((b1 & 0x0F) << 4));
    }
    __threadfence_block();
    bool any_diamond = __any_sync(0xFFFFFFFFu, has_diamond);
    __syncwarp();

    if (!any_diamond && lane == 0) {
        for (int attempts = 0; attempts < 1000; attempts++) {
            int r = (int)(philox_u32_at(seed, rng_subseq, WG_DIAMOND_OFF + 2 * (uint64_t)attempts) % MAP_SIZE);
            int c = (int)(philox_u32_at(seed, rng_subseq, WG_DIAMOND_OFF + 2 * (uint64_t)attempts + 1) % MAP_SIZE);
            if (map_get(map, r, c) == BLK_STONE) {
                map_set(map, r, c, BLK_DIAMOND);
                break;
            }
        }
    }

    if (lane == 0)
        wg_reset_fields(g, e, seed, rng_subseq);
    __syncwarp();
}

// ============================================================
// Step Logic
// ============================================================
__device__ void do_crafting(EnvSoA& g, int e, int action) {
    const int n = g.n;
    const uint8_t* __restrict__ map = env_map(g, e);
    int pr = g.player_r[e], pc = g.player_c[e];
    bool near_table = is_near_block(map, pr, pc, BLK_TABLE);
    bool near_furnace = is_near_block(map, pr, pc, BLK_FURNACE);

    if (action == ACT_MAKE_WOOD_PICK && near_table && g.inv[0*n+e] >= 1) {
        g.inv[0*n+e]--; g.inv[6*n+e]++; g.ach[e] |= 1u << ACH_MAKE_WOOD_PICK;
    }
    if (action == ACT_MAKE_STONE_PICK && near_table && g.inv[0*n+e] >= 1 && g.inv[1*n+e] >= 1) {
        g.inv[0*n+e]--; g.inv[1*n+e]--; g.inv[7*n+e]++; g.ach[e] |= 1u << ACH_MAKE_STONE_PICK;
    }
    if (action == ACT_MAKE_IRON_PICK && near_table && near_furnace &&
        g.inv[0*n+e] >= 1 && g.inv[1*n+e] >= 1 && g.inv[3*n+e] >= 1 && g.inv[2*n+e] >= 1) {
        g.inv[0*n+e]--; g.inv[1*n+e]--; g.inv[3*n+e]--; g.inv[2*n+e]--;
        g.inv[8*n+e]++; g.ach[e] |= 1u << ACH_MAKE_IRON_PICK;
    }
    if (action == ACT_MAKE_WOOD_SWORD && near_table && g.inv[0*n+e] >= 1) {
        g.inv[0*n+e]--; g.inv[9*n+e]++; g.ach[e] |= 1u << ACH_MAKE_WOOD_SWORD;
    }
    if (action == ACT_MAKE_STONE_SWORD && near_table && g.inv[0*n+e] >= 1 && g.inv[1*n+e] >= 1) {
        g.inv[0*n+e]--; g.inv[1*n+e]--; g.inv[10*n+e]++; g.ach[e] |= 1u << ACH_MAKE_STONE_SWORD;
    }
    if (action == ACT_MAKE_IRON_SWORD && near_table && near_furnace &&
        g.inv[0*n+e] >= 1 && g.inv[1*n+e] >= 1 && g.inv[3*n+e] >= 1 && g.inv[2*n+e] >= 1) {
        g.inv[0*n+e]--; g.inv[1*n+e]--; g.inv[3*n+e]--; g.inv[2*n+e]--;
        g.inv[11*n+e]++; g.ach[e] |= 1u << ACH_MAKE_IRON_SWORD;
    }
}

__device__ void do_action(EnvSoA& g, int e, curandStatePhilox4_32_10_t* rng) {
    const int n = g.n;
    int tr = g.player_r[e] + DIR_DR[g.player_dir[e]];
    int tc = g.player_c[e] + DIR_DC[g.player_dir[e]];
    if (!in_bounds(tr, tc)) return;

    int dmg = get_damage(g, e);
    bool attacked = false;

    for (int i = 0; i < MAX_ZOMBIES && !attacked; i++) {
        if (g.zombie_mask[i*n+e] && g.zombie_r[i*n+e] == tr && g.zombie_c[i*n+e] == tc) {
            g.zombie_hp[i*n+e] -= dmg;
            if (g.zombie_hp[i*n+e] <= 0) {
                g.zombie_mask[i*n+e] = 0;
                g.ach[e] |= 1u << ACH_DEFEAT_ZOMBIE;
            }
            attacked = true;
        }
    }
    for (int i = 0; i < MAX_COWS && !attacked; i++) {
        if (g.cow_mask[i*n+e] && g.cow_r[i*n+e] == tr && g.cow_c[i*n+e] == tc) {
            g.cow_hp[i*n+e] -= dmg;
            if (g.cow_hp[i*n+e] <= 0) {
                g.cow_mask[i*n+e] = 0;
                g.ach[e] |= 1u << ACH_EAT_COW;
                g.food[e] = min_i(9, g.food[e] + 6);
                g.hunger[e] = 0;
            }
            attacked = true;
        }
    }
    for (int i = 0; i < MAX_SKELETONS && !attacked; i++) {
        if (g.skel_mask[i*n+e] && g.skel_r[i*n+e] == tr && g.skel_c[i*n+e] == tc) {
            g.skel_hp[i*n+e] -= dmg;
            if (g.skel_hp[i*n+e] <= 0) {
                g.skel_mask[i*n+e] = 0;
                g.ach[e] |= 1u << ACH_DEFEAT_SKELETON;
            }
            attacked = true;
        }
    }

    if (attacked) return;

    uint8_t* __restrict__ map = env_map(g, e);
    int8_t blk = map_get(map, tr, tc);
    switch (blk) {
        case BLK_TREE:
            map_set(map, tr, tc, BLK_GRASS);
            g.inv[0*n+e] = min_i(9, g.inv[0*n+e] + 1);
            g.ach[e] |= 1u << ACH_COLLECT_WOOD;
            break;
        case BLK_STONE:
            if (g.inv[6*n+e] > 0 || g.inv[7*n+e] > 0 || g.inv[8*n+e] > 0) {
                map_set(map, tr, tc, BLK_PATH);
                g.inv[1*n+e] = min_i(9, g.inv[1*n+e] + 1);
                g.ach[e] |= 1u << ACH_COLLECT_STONE;
            }
            break;
        case BLK_COAL:
            if (g.inv[6*n+e] > 0 || g.inv[7*n+e] > 0 || g.inv[8*n+e] > 0) {
                map_set(map, tr, tc, BLK_PATH);
                g.inv[2*n+e] = min_i(9, g.inv[2*n+e] + 1);
                g.ach[e] |= 1u << ACH_COLLECT_COAL;
            }
            break;
        case BLK_IRON:
            if (g.inv[7*n+e] > 0 || g.inv[8*n+e] > 0) {
                map_set(map, tr, tc, BLK_PATH);
                g.inv[3*n+e] = min_i(9, g.inv[3*n+e] + 1);
                g.ach[e] |= 1u << ACH_COLLECT_IRON;
            }
            break;
        case BLK_DIAMOND:
            if (g.inv[8*n+e] > 0) {
                map_set(map, tr, tc, BLK_PATH);
                g.inv[4*n+e] = min_i(9, g.inv[4*n+e] + 1);
                g.ach[e] |= 1u << ACH_COLLECT_DIAMOND;
            }
            break;
        case BLK_GRASS:
            if (rand_f(rng) < 0.1f) {
                g.inv[5*n+e] = min_i(9, g.inv[5*n+e] + 1);
                g.ach[e] |= 1u << ACH_COLLECT_SAPLING;
            }
            break;
        case BLK_WATER:
            g.drink[e] = min_i(9, g.drink[e] + 1);
            g.thirst[e] = 0;
            g.ach[e] |= 1u << ACH_COLLECT_DRINK;
            break;
        case BLK_RIPE_PLANT:
            map_set(map, tr, tc, BLK_PLANT);
            g.food[e] = min_i(9, g.food[e] + 4);
            g.hunger[e] = 0;
            g.ach[e] |= 1u << ACH_EAT_PLANT;
            for (int i = 0; i < MAX_PLANTS; i++) {
                if (g.plant_mask[i*n+e] && g.plant_r[i*n+e] == tr && g.plant_c[i*n+e] == tc) {
                    g.plant_age[i*n+e] = 0;
                    break;
                }
            }
            break;
    }
}

__device__ void place_block(EnvSoA& g, int e, int action) {
    const int n = g.n;
    int tr = g.player_r[e] + DIR_DR[g.player_dir[e]];
    int tc = g.player_c[e] + DIR_DC[g.player_dir[e]];
    if (!in_bounds(tr, tc)) return;
    if (has_mob_at(g, e, tr, tc)) return;

    uint8_t* __restrict__ map = env_map(g, e);
    int8_t blk = map_get(map, tr, tc);

    if (action == ACT_PLACE_TABLE && g.inv[0*n+e] >= 2 && !is_solid(blk)) {
        map_set(map, tr, tc, BLK_TABLE); g.inv[0*n+e] -= 2;
        g.ach[e] |= 1u << ACH_PLACE_TABLE;
    }
    else if (action == ACT_PLACE_FURNACE && g.inv[1*n+e] >= 1 && !is_solid(blk)) {
        map_set(map, tr, tc, BLK_FURNACE); g.inv[1*n+e] -= 1;
        g.ach[e] |= 1u << ACH_PLACE_FURNACE;
    }
    else if (action == ACT_PLACE_STONE && g.inv[1*n+e] >= 1 && (!is_solid(blk) || blk == BLK_WATER)) {
        map_set(map, tr, tc, BLK_STONE); g.inv[1*n+e] -= 1;
        g.ach[e] |= 1u << ACH_PLACE_STONE;
    }
    else if (action == ACT_PLACE_PLANT && g.inv[5*n+e] >= 1 && blk == BLK_GRASS) {
        map_set(map, tr, tc, BLK_PLANT); g.inv[5*n+e] -= 1;
        g.ach[e] |= 1u << ACH_PLACE_PLANT;
        for (int i = 0; i < MAX_PLANTS; i++) {
            if (!g.plant_mask[i*n+e]) {
                g.plant_r[i*n+e] = tr; g.plant_c[i*n+e] = tc;
                g.plant_age[i*n+e] = 0; g.plant_mask[i*n+e] = 1;
                break;
            }
        }
    }
}

__device__ void move_player(EnvSoA& g, int e, int action) {
    if (action < 1 || action > 4) return;
    int nr = g.player_r[e] + DIR_DR[action];
    int nc = g.player_c[e] + DIR_DC[action];
    g.player_dir[e] = action;
    if (!in_bounds(nr, nc)) return;
    if (is_solid(map_get(env_map(g, e), nr, nc))) return;
    if (has_mob_at(g, e, nr, nc)) return;
    g.player_r[e] = nr; g.player_c[e] = nc;
}

__device__ bool can_move_mob(const EnvSoA& g, int e, int r, int c) {
    if (!in_bounds(r, c)) return false;
    int8_t blk = map_get(env_map(g, e), r, c);
    if (is_solid(blk)) return false;
    if (blk == BLK_LAVA) return false;
    if (has_mob_at(g, e, r, c)) return false;
    if (r == g.player_r[e] && c == g.player_c[e]) return false;
    return true;
}

__device__ void update_mobs(EnvSoA& g, int e, curandStatePhilox4_32_10_t* rng) {
    const int n = g.n;
    int pr = g.player_r[e], pc = g.player_c[e];

    for (int i = 0; i < MAX_ZOMBIES; i++) {
        if (!g.zombie_mask[i*n+e]) continue;
        int zr = g.zombie_r[i*n+e], zc = g.zombie_c[i*n+e];
        int dist = l1_dist(zr, zc, pr, pc);

        if (dist >= MOB_DESPAWN_DIST) { g.zombie_mask[i*n+e] = 0; continue; }

        if (dist <= 1 && g.zombie_cd[i*n+e] <= 0) {
            int dmg = g.is_sleeping[e] ? 7 : 2;
            g.health[e] -= dmg;
            g.zombie_cd[i*n+e] = 5;
            g.is_sleeping[e] = 0;
        }
        g.zombie_cd[i*n+e] = max_i(0, g.zombie_cd[i*n+e] - 1);

        int dr = 0, dc = 0;
        if (dist < 10 && rand_f(rng) < 0.75f) {
            int adr = abs(pr - zr), adc = abs(pc - zc);
            if (adr > adc || (adr == adc && rand_f(rng) < 0.5f))
                dr = sign_i(pr - zr);
            else
                dc = sign_i(pc - zc);
        } else {
            int d = rand_int(rng, 4);
            dr = DIR_DR[d+1]; dc = DIR_DC[d+1];
        }
        int nr = zr + dr, nc = zc + dc;
        if (can_move_mob(g, e, nr, nc)) {
            g.zombie_r[i*n+e] = nr; g.zombie_c[i*n+e] = nc;
        }
    }

    for (int i = 0; i < MAX_COWS; i++) {
        if (!g.cow_mask[i*n+e]) continue;
        int cr = g.cow_r[i*n+e], cc = g.cow_c[i*n+e];
        int dist = l1_dist(cr, cc, pr, pc);

        if (dist >= MOB_DESPAWN_DIST) { g.cow_mask[i*n+e] = 0; continue; }

        int d = rand_int(rng, 8);
        if (d < 4) {
            int dr = DIR_DR[d+1], dc2 = DIR_DC[d+1];
            int nr = cr + dr, nc = cc + dc2;
            if (can_move_mob(g, e, nr, nc)) {
                g.cow_r[i*n+e] = nr; g.cow_c[i*n+e] = nc;
            }
        }
    }

    for (int i = 0; i < MAX_SKELETONS; i++) {
        if (!g.skel_mask[i*n+e]) continue;
        int sr = g.skel_r[i*n+e], sc = g.skel_c[i*n+e];
        int dist = l1_dist(sr, sc, pr, pc);

        if (dist >= MOB_DESPAWN_DIST) { g.skel_mask[i*n+e] = 0; continue; }

        if (dist >= 4 && dist <= 5 && g.skel_cd[i*n+e] <= 0) {
            for (int a = 0; a < MAX_ARROWS; a++) {
                if (!g.arrow_mask[a*n+e]) {
                    g.arrow_mask[a*n+e] = 1;
                    g.arrow_r[a*n+e] = sr; g.arrow_c[a*n+e] = sc;
                    int adr = abs(pr - sr), adc = abs(pc - sc);
                    g.arrow_dr[a*n+e] = (adr > 0) ? sign_i(pr - sr) : 0;
                    g.arrow_dc[a*n+e] = (adc > 0) ? sign_i(pc - sc) : 0;
                    break;
                }
            }
            g.skel_cd[i*n+e] = 4;
        }
        g.skel_cd[i*n+e] = max_i(0, g.skel_cd[i*n+e] - 1);

        int dr = 0, dc = 0;
        bool random_move = rand_f(rng) < 0.15f;
        if (!random_move) {
            if (dist >= 10) {
                int adr = abs(pr - sr), adc = abs(pc - sc);
                if (adr > adc || (adr == adc && rand_f(rng) < 0.5f))
                    dr = sign_i(pr - sr);
                else
                    dc = sign_i(pc - sc);
            } else if (dist <= 3) {
                int adr = abs(pr - sr), adc = abs(pc - sc);
                if (adr > adc || (adr == adc && rand_f(rng) < 0.5f))
                    dr = -sign_i(pr - sr);
                else
                    dc = -sign_i(pc - sc);
            } else {
                random_move = true;
            }
        }
        if (random_move) {
            int d = rand_int(rng, 4);
            dr = DIR_DR[d+1]; dc = DIR_DC[d+1];
        }
        int nr = sr + dr, nc = sc + dc;
        if (can_move_mob(g, e, nr, nc)) {
            g.skel_r[i*n+e] = nr; g.skel_c[i*n+e] = nc;
        }
    }

    for (int i = 0; i < MAX_ARROWS; i++) {
        if (!g.arrow_mask[i*n+e]) continue;
        int nr = g.arrow_r[i*n+e] + g.arrow_dr[i*n+e];
        int nc = g.arrow_c[i*n+e] + g.arrow_dc[i*n+e];

        if (!in_bounds(nr, nc)) { g.arrow_mask[i*n+e] = 0; continue; }
        uint8_t* __restrict__ map = env_map(g, e);
        int8_t blk = map_get(map, nr, nc);
        if (is_solid(blk) && blk != BLK_WATER) {
            if (blk == BLK_FURNACE || blk == BLK_TABLE) map_set(map, nr, nc, BLK_PATH);
            g.arrow_mask[i*n+e] = 0;
            continue;
        }
        if (nr == pr && nc == pc) {
            g.health[e] -= 2;
            g.is_sleeping[e] = 0;
            g.arrow_mask[i*n+e] = 0;
            continue;
        }
        g.arrow_r[i*n+e] = nr; g.arrow_c[i*n+e] = nc;
    }
}

__device__ void spawn_mobs(EnvSoA& g, int e, curandStatePhilox4_32_10_t* rng) {
    const int n = g.n;
    int pr = g.player_r[e], pc = g.player_c[e];
    const uint8_t* __restrict__ map = env_map(g, e);

    int n_cows = 0, n_zombies = 0, n_skels = 0;
    for (int i = 0; i < MAX_COWS; i++) n_cows += g.cow_mask[i*n+e];
    for (int i = 0; i < MAX_ZOMBIES; i++) n_zombies += g.zombie_mask[i*n+e];
    for (int i = 0; i < MAX_SKELETONS; i++) n_skels += g.skel_mask[i*n+e];

    auto try_spawn = [&](int min_dist, int max_dist, bool need_grass, bool need_path, int* out_r, int* out_c) -> bool {
        for (int attempts = 0; attempts < 20; attempts++) {
            int r = rand_int(rng, MAP_SIZE);
            int c = rand_int(rng, MAP_SIZE);
            int dist = l1_dist(r, c, pr, pc);
            if (dist < min_dist || dist >= max_dist) continue;
            if (has_mob_at(g, e, r, c)) continue;
            if (r == pr && c == pc) continue;
            int8_t blk = map_get(map, r, c);
            if (need_grass && blk != BLK_GRASS) continue;
            if (need_path && blk != BLK_PATH) continue;
            if (!need_grass && !need_path && blk != BLK_GRASS && blk != BLK_PATH) continue;
            *out_r = r; *out_c = c;
            return true;
        }
        return false;
    };

    if (n_cows < MAX_COWS && rand_f(rng) < 0.1f) {
        int r, c;
        if (try_spawn(3, MOB_DESPAWN_DIST, true, false, &r, &c)) {
            for (int i = 0; i < MAX_COWS; i++) {
                if (!g.cow_mask[i*n+e]) {
                    g.cow_mask[i*n+e] = 1; g.cow_r[i*n+e] = r; g.cow_c[i*n+e] = c; g.cow_hp[i*n+e] = 3;
                    break;
                }
            }
        }
    }

    float zombie_chance = 0.02f + 0.1f * (1.0f - g.light_level[e]) * (1.0f - g.light_level[e]);
    if (n_zombies < MAX_ZOMBIES && rand_f(rng) < zombie_chance) {
        int r, c;
        if (try_spawn(9, MOB_DESPAWN_DIST, false, false, &r, &c)) {
            for (int i = 0; i < MAX_ZOMBIES; i++) {
                if (!g.zombie_mask[i*n+e]) {
                    g.zombie_mask[i*n+e] = 1; g.zombie_r[i*n+e] = r; g.zombie_c[i*n+e] = c;
                    g.zombie_hp[i*n+e] = 5; g.zombie_cd[i*n+e] = 0;
                    break;
                }
            }
        }
    }

    if (n_skels < MAX_SKELETONS && rand_f(rng) < 0.05f) {
        int r, c;
        if (try_spawn(9, MOB_DESPAWN_DIST, false, true, &r, &c)) {
            for (int i = 0; i < MAX_SKELETONS; i++) {
                if (!g.skel_mask[i*n+e]) {
                    g.skel_mask[i*n+e] = 1; g.skel_r[i*n+e] = r; g.skel_c[i*n+e] = c;
                    g.skel_hp[i*n+e] = 3; g.skel_cd[i*n+e] = 0;
                    break;
                }
            }
        }
    }
}

__device__ void update_plants(EnvSoA& g, int e) {
    const int n = g.n;
    for (int i = 0; i < MAX_PLANTS; i++) {
        if (!g.plant_mask[i*n+e]) continue;
        g.plant_age[i*n+e]++;
        if (g.plant_age[i*n+e] >= 600) {
            int r = g.plant_r[i*n+e], c = g.plant_c[i*n+e];
            uint8_t* __restrict__ map = env_map(g, e);
            if (in_bounds(r, c) && map_get(map, r, c) == BLK_PLANT) {
                map_set(map, r, c, BLK_RIPE_PLANT);
            }
        }
    }
}

__device__ void update_intrinsics(EnvSoA& g, int e, int action) {
    if (action == ACT_SLEEP && g.energy[e] < 9) g.is_sleeping[e] = 1;
    if (g.energy[e] >= 9 && g.is_sleeping[e]) {
        g.is_sleeping[e] = 0;
        g.ach[e] |= 1u << ACH_WAKE_UP;
    }

    float sleep_mul = g.is_sleeping[e] ? 0.5f : 1.0f;

    g.hunger[e] += sleep_mul;
    if (g.hunger[e] > 25.0f) { g.food[e]--; g.hunger[e] = 0; }

    g.thirst[e] += sleep_mul;
    if (g.thirst[e] > 20.0f) { g.drink[e]--; g.thirst[e] = 0; }

    if (g.is_sleeping[e]) g.fatigue[e] -= 1.0f;
    else g.fatigue[e] += 1.0f;
    if (g.fatigue[e] > 30.0f) { g.energy[e]--; g.fatigue[e] = 0; }
    if (g.fatigue[e] < -10.0f) { g.energy[e] = min_i(g.energy[e] + 1, 9); g.fatigue[e] = 0; }

    bool all_needs = (g.food[e] > 0) && (g.drink[e] > 0) && (g.energy[e] > 0 || g.is_sleeping[e]);
    if (all_needs) g.recover[e] += g.is_sleeping[e] ? 2.0f : 1.0f;
    else g.recover[e] += g.is_sleeping[e] ? -0.5f : -1.0f;
    if (g.recover[e] > 25.0f) { g.health[e] = min_i(g.health[e] + 1, 9); g.recover[e] = 0; }
    if (g.recover[e] < -15.0f) { g.health[e]--; g.recover[e] = 0; }
}

// ============================================================
// Observation
// ============================================================
// Gather the 63-tile view: block ids plus a 4-bit mob mask per cell.
// Mobs are scanned once each and stamped into their view cell (instead
// of scanning every mob for every cell); emitted values are identical.
__device__ __forceinline__ void gather_view(
    const EnvSoA& g, int e, int8_t* view_blk, uint8_t* view_mob
) {
    const int n = g.n;
    const uint8_t* __restrict__ map = env_map(g, e);
    int pr = g.player_r[e], pc = g.player_c[e];

    for (int dr = -3; dr <= 3; dr++) {
        for (int dc = -4; dc <= 4; dc++) {
            int r = pr + dr, c = pc + dc;
            int cell = (dr + 3) * OBS_MAP_COLS + (dc + 4);
            view_blk[cell] = in_bounds(r, c) ? map_get(map, r, c) : BLK_OUT_OF_BOUNDS;
            view_mob[cell] = 0;
        }
    }

    auto stamp = [&](int r, int c, uint8_t bit) {
        int dr = r - pr, dc = c - pc;
        if (dr < -3 || dr > 3 || dc < -4 || dc > 4) return;
        view_mob[(dr + 3) * OBS_MAP_COLS + (dc + 4)] |= bit;
    };

    for (int i = 0; i < MAX_ZOMBIES; i++)
        if (g.zombie_mask[i*n+e]) stamp(g.zombie_r[i*n+e], g.zombie_c[i*n+e], 1);
    for (int i = 0; i < MAX_COWS; i++)
        if (g.cow_mask[i*n+e]) stamp(g.cow_r[i*n+e], g.cow_c[i*n+e], 2);
    for (int i = 0; i < MAX_SKELETONS; i++)
        if (g.skel_mask[i*n+e]) stamp(g.skel_r[i*n+e], g.skel_c[i*n+e], 4);
    for (int i = 0; i < MAX_ARROWS; i++)
        if (g.arrow_mask[i*n+e]) stamp(g.arrow_r[i*n+e], g.arrow_c[i*n+e], 8);
}

__device__ void build_observation(const EnvSoA& g, int e, float* obs) {
    const int n = g.n;
    int8_t view_blk[OBS_MAP_CELLS];
    uint8_t view_mob[OBS_MAP_CELLS];
    gather_view(g, e, view_blk, view_mob);

    int obs_idx = 0;
    for (int cell = 0; cell < OBS_MAP_CELLS; cell++) {
        int8_t blk = view_blk[cell];
        for (int b = 0; b < NUM_BLOCK_TYPES; b++)
            obs[obs_idx++] = (blk == b) ? 1.0f : 0.0f;
        uint8_t m = view_mob[cell];
        obs[obs_idx++] = (m & 1) ? 1.0f : 0.0f;
        obs[obs_idx++] = (m & 2) ? 1.0f : 0.0f;
        obs[obs_idx++] = (m & 4) ? 1.0f : 0.0f;
        obs[obs_idx++] = (m & 8) ? 1.0f : 0.0f;
    }

    for (int i = 0; i < NUM_INVENTORY; i++)
        obs[obs_idx++] = (float)g.inv[i*n+e] / 10.0f;

    obs[obs_idx++] = (float)g.health[e] / 10.0f;
    obs[obs_idx++] = (float)g.food[e] / 10.0f;
    obs[obs_idx++] = (float)g.drink[e] / 10.0f;
    obs[obs_idx++] = (float)g.energy[e] / 10.0f;

    for (int d = 1; d <= 4; d++)
        obs[obs_idx++] = (g.player_dir[e] == d) ? 1.0f : 0.0f;

    obs[obs_idx++] = g.light_level[e];
    obs[obs_idx++] = g.is_sleeping[e] ? 1.0f : 0.0f;
}

// Compact uint8 observation (see OBS_DIM_COMPACT layout in craftax.cuh).
// 148 bytes vs 5380 -- 36x less obs traffic. expand_obs_kernel reproduces
// the float observation bit-for-bit.
__device__ void build_observation_compact(const EnvSoA& g, int e, uint8_t* obs) {
    const int n = g.n;
    int8_t view_blk[OBS_MAP_CELLS];
    uint8_t view_mob[OBS_MAP_CELLS];
    gather_view(g, e, view_blk, view_mob);

    for (int cell = 0; cell < OBS_MAP_CELLS; cell++) {
        obs[cell] = (uint8_t)view_blk[cell];
        obs[OBS_MAP_CELLS + cell] = view_mob[cell];
    }

    int idx = 2 * OBS_MAP_CELLS;
    for (int i = 0; i < NUM_INVENTORY; i++)
        obs[idx++] = (uint8_t)g.inv[i*n+e];
    obs[idx++] = (uint8_t)g.health[e];
    obs[idx++] = (uint8_t)g.food[e];
    obs[idx++] = (uint8_t)g.drink[e];
    obs[idx++] = (uint8_t)g.energy[e];
    obs[idx++] = (uint8_t)g.player_dir[e];
    obs[idx++] = g.is_sleeping[e] ? 1 : 0;
    float light = g.light_level[e];
    memcpy(obs + idx, &light, sizeof(float));
}

// Expand compact obs back to the exact 1345-float observation.
// One thread per (env, cell-ish chunk): simple env-per-thread version.
extern "C" __global__ void expand_obs_kernel(
    const uint8_t* __restrict__ compact, float* __restrict__ obs, int num_envs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;

    const uint8_t* __restrict__ c = compact + (size_t)idx * OBS_DIM_COMPACT;
    float* __restrict__ o = obs + (size_t)idx * OBS_DIM;

    int obs_idx = 0;
    for (int cell = 0; cell < OBS_MAP_CELLS; cell++) {
        int8_t blk = (int8_t)c[cell];
        for (int b = 0; b < NUM_BLOCK_TYPES; b++)
            o[obs_idx++] = (blk == b) ? 1.0f : 0.0f;
        uint8_t m = c[OBS_MAP_CELLS + cell];
        o[obs_idx++] = (m & 1) ? 1.0f : 0.0f;
        o[obs_idx++] = (m & 2) ? 1.0f : 0.0f;
        o[obs_idx++] = (m & 4) ? 1.0f : 0.0f;
        o[obs_idx++] = (m & 8) ? 1.0f : 0.0f;
    }

    int idx2 = 2 * OBS_MAP_CELLS;
    for (int i = 0; i < NUM_INVENTORY; i++)
        o[obs_idx++] = (float)(int8_t)c[idx2++] / 10.0f;
    o[obs_idx++] = (float)(int8_t)c[idx2++] / 10.0f;
    o[obs_idx++] = (float)(int8_t)c[idx2++] / 10.0f;
    o[obs_idx++] = (float)(int8_t)c[idx2++] / 10.0f;
    o[obs_idx++] = (float)(int8_t)c[idx2++] / 10.0f;
    int8_t dir = (int8_t)c[idx2++];
    for (int d = 1; d <= 4; d++)
        o[obs_idx++] = (dir == d) ? 1.0f : 0.0f;
    uint8_t sleeping = c[idx2++];
    float light;
    memcpy(&light, c + idx2, sizeof(float));
    o[obs_idx++] = light;
    o[obs_idx++] = sleeping ? 1.0f : 0.0f;
}

// ============================================================
// Step core (shared by fused and split kernels)
// ============================================================
__device__ __forceinline__ bool step_env(
    EnvSoA& g, int e, int action_in, float* __restrict__ rewards
) {
    const int n = g.n;
    curandStatePhilox4_32_10_t rng = g.rng[e];

    int action = action_in;
    int old_health = g.health[e];
    uint32_t old_ach = g.ach[e];

    if (g.is_sleeping[e]) action = ACT_NOOP;

    do_crafting(g, e, action);
    if (action == ACT_DO) do_action(g, e, &rng);
    if (action >= ACT_PLACE_STONE && action <= ACT_PLACE_PLANT) place_block(g, e, action);
    move_player(g, e, action);
    update_mobs(g, e, &rng);
    spawn_mobs(g, e, &rng);
    update_plants(g, e);
    update_intrinsics(g, e, action_in);

    for (int i = 0; i < NUM_INVENTORY; i++)
        g.inv[i*n+e] = clamp_i(g.inv[i*n+e], 0, 9);

    int32_t t = ++g.timestep[e];
    float t_frac = fmodf((float)t / (float)DAY_LENGTH, 1.0f) + 0.3f;
    float cos_val = __cosf(3.14159265f * t_frac);
    g.light_level[e] = 1.0f - fabsf(cos_val * cos_val * cos_val);

    float ach_reward = (float)__popc(g.ach[e] & ~old_ach);
    float health_reward = (float)(g.health[e] - old_health) * 0.1f;
    rewards[e] = ach_reward + health_reward;

    bool done = (t >= MAX_TIMESTEPS) || (g.health[e] <= 0);
    int pr = g.player_r[e], pc = g.player_c[e];
    if (in_bounds(pr, pc) && map_get(env_map(g, e), pr, pc) == BLK_LAVA)
        done = true;

    g.rng[e] = rng;
    return done;
}

// ============================================================
// Main Kernels
// ============================================================
extern "C" __global__ void reset_kernel(
    EnvSoA g, float* obs, uint8_t* obs_compact, int obs_mode,
    int num_envs, uint64_t seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;

    generate_world(g, idx, seed, idx);
    if (obs_mode != 1)
        build_observation(g, idx, obs + (size_t)idx * OBS_DIM);
    if (obs_mode >= 1)
        build_observation_compact(g, idx, obs_compact + (size_t)idx * OBS_DIM_COMPACT);
}

// Step kernel: game logic only, marks dones but does NOT auto-reset
extern "C" __global__ void step_only_kernel(
    EnvSoA g, const int32_t* __restrict__ actions,
    float* __restrict__ rewards, int8_t* __restrict__ dones,
    int num_envs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;
    dones[idx] = step_env(g, idx, actions[idx], rewards) ? 1 : 0;
}

// Auto-reset kernel: only runs on done envs, then builds obs for ALL envs
extern "C" __global__ void autoreset_obs_kernel(
    EnvSoA g, const int8_t* __restrict__ dones,
    float* obs, uint8_t* obs_compact, int obs_mode,
    int num_envs, uint64_t reset_seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;

    if (dones[idx]) {
        generate_world(g, idx, reset_seed, idx + num_envs);
    }
    if (obs_mode != 1)
        build_observation(g, idx, obs + (size_t)idx * OBS_DIM);
    if (obs_mode >= 1)
        build_observation_compact(g, idx, obs_compact + (size_t)idx * OBS_DIM_COMPACT);
}

// ============================================================
// Split-reset path: step marks dones into a compact list, a
// warp-cooperative kernel regenerates only the done envs, then a
// uniform kernel builds obs for everyone. reset_ctrl is 2 ints:
// [0] = done count (filled by step_mark), [1] = work cursor
// (used by reset_warp). Zero both before step_mark each step.
// ============================================================
extern "C" __global__ void step_mark_kernel(
    EnvSoA g, const int32_t* __restrict__ actions,
    float* __restrict__ rewards, int8_t* __restrict__ dones,
    int32_t* __restrict__ reset_list, int32_t* __restrict__ reset_ctrl,
    int num_envs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;
    bool done = step_env(g, idx, actions[idx], rewards);
    dones[idx] = done ? 1 : 0;
    if (done) {
        int slot = atomicAdd(&reset_ctrl[0], 1);
        reset_list[slot] = idx;
    }
}

// One warp per resetting env, work-stealing over the compact list.
// Launch with a fixed grid (RESET_WARP_BLOCK threads/block); warps
// loop until the list is drained, so grid size is independent of
// the (device-side) done count.
#define RESET_WARP_BLOCK 128
extern "C" __global__ void reset_warp_kernel(
    EnvSoA g, const int32_t* __restrict__ reset_list,
    int32_t* __restrict__ reset_ctrl,
    int num_envs, uint64_t reset_seed
) {
    __shared__ float angles[RESET_WARP_BLOCK / 32][WG_ANGLE_COUNT];
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    for (;;) {
        int i = 0;
        if (lane == 0) i = atomicAdd(&reset_ctrl[1], 1);
        i = __shfl_sync(0xFFFFFFFFu, i, 0);
        if (i >= reset_ctrl[0]) return;
        int e = reset_list[i];
        generate_world_warp(g, e, reset_seed, e + num_envs, angles[warp_in_block]);
    }
}

// Warp-cooperative full reset (initial reset of all envs).
// Launch num_envs warps; subseq = env index, matching reset_kernel.
extern "C" __global__ void reset_all_warp_kernel(
    EnvSoA g, int num_envs, uint64_t seed
) {
    __shared__ float angles[RESET_WARP_BLOCK / 32][WG_ANGLE_COUNT];
    const int warp_in_block = threadIdx.x >> 5;
    int e = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (e >= num_envs) return;
    generate_world_warp(g, e, seed, e, angles[warp_in_block]);
}

// Uniform obs pass, run after resets so every env sees fresh state.
extern "C" __global__ void obs_kernel(
    EnvSoA g, float* obs, uint8_t* obs_compact, int obs_mode, int num_envs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;
    if (obs_mode != 1)
        build_observation(g, idx, obs + (size_t)idx * OBS_DIM);
    if (obs_mode >= 1)
        build_observation_compact(g, idx, obs_compact + (size_t)idx * OBS_DIM_COMPACT);
}

// Combined step kernel (for backward compat)
extern "C" __global__ void step_kernel(
    EnvSoA g, const int32_t* __restrict__ actions,
    float* obs, uint8_t* obs_compact, int obs_mode,
    float* __restrict__ rewards, int8_t* __restrict__ dones,
    int num_envs, uint64_t reset_seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;

    bool done = step_env(g, idx, actions[idx], rewards);
    dones[idx] = done ? 1 : 0;

    if (done) {
        generate_world(g, idx, reset_seed, idx + num_envs);
    }
    if (obs_mode != 1)
        build_observation(g, idx, obs + (size_t)idx * OBS_DIM);
    if (obs_mode >= 1)
        build_observation_compact(g, idx, obs_compact + (size_t)idx * OBS_DIM_COMPACT);
}
