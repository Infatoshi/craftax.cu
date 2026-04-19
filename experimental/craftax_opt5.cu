// opt5: warp-per-env with shared-memory state staging.
//
// Layout per block: BLOCK_SIZE = WARPS_PER_BLOCK * 32 threads.
// Each warp handles one env. Per-env state + map live in shared memory for the
// duration of the kernel. Bulk load from global -> shared (warp-coalesced),
// game logic on lane 0 of each warp, obs build lane-parallel, bulk store back.
//
// Bitwise-identical outputs to Oracle (same code paths, just relocated in space).
#include "craftax_opt5.cuh"
#include <cmath>

static_assert(sizeof(EnvState) % 16 == 0, "EnvState must be 16B-aligned for int4 I/O");
static_assert(offsetof(EnvState, map_packed) == 0, "map_packed must be at offset 0");
static_assert(MAP_PACKED_SIZE % 16 == 0, "map_packed must be int4-sized");

constexpr int WARPS_PER_BLOCK = 4;
constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * 32;
constexpr int ENVSTATE_U32 = sizeof(EnvState) / 4;
constexpr int ENVSTATE_I4  = sizeof(EnvState) / 16;
constexpr int MAP_I4       = MAP_PACKED_SIZE / 16;  // int4s in the map region

__device__ __forceinline__ void warp_bulk_copy_i4(int4* __restrict__ dst, const int4* __restrict__ src, int lane) {
    #pragma unroll 2
    for (int i = lane; i < ENVSTATE_I4; i += 32) dst[i] = src[i];
}

// Write only the non-map portion of the state (offset MAP_PACKED_SIZE onward).
__device__ __forceinline__ void warp_bulk_copy_nonmap_i4(int4* __restrict__ dst, const int4* __restrict__ src, int lane) {
    #pragma unroll 2
    for (int i = lane + MAP_I4; i < ENVSTATE_I4; i += 32) dst[i] = src[i];
}

// --- Helpers copied from opt; operate on any EnvState& (shared or global) ---

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

__device__ void generate_world(EnvState& s, uint64_t seed, int env_id) {
    curand_init(seed, env_id, 0, &s.rng);
    const int GRID = 10;
    float angles[4][GRID * GRID];
    for (int layer = 0; layer < 4; layer++)
        for (int i = 0; i < GRID * GRID; i++)
            angles[layer][i] = rand_f(&s.rng) * 2.0f * 3.14159265f;
    float scale = (float)MAP_SIZE / (float)(GRID - 1);
    int center = MAP_SIZE / 2;
    bool has_diamond = false;
    for (int r = 0; r < MAP_SIZE; r++) {
        for (int c = 0; c < MAP_SIZE; c++) {
            float nr = (float)r / scale, nc = (float)c / scale;
            float wn = perlin_2d(nr, nc, angles[0], GRID);
            float mn = perlin_2d(nr, nc, angles[1], GRID);
            float tn = perlin_2d(nr, nc, angles[2], GRID);
            float pn = perlin_2d(nr, nc, angles[3], GRID);
            float dist = sqrtf((float)((r-center)*(r-center) + (c-center)*(c-center)));
            float prox = 1.0f - min_f(dist / 20.0f, 1.0f);
            float wv = wn - prox * 0.3f, mv = mn - prox * 0.3f;
            int8_t blk = BLK_GRASS;
            if (wv > 0.7f) blk = BLK_WATER;
            else if (wv > 0.6f && wv <= 0.75f) blk = BLK_SAND;
            else if (mv > 0.7f) {
                blk = BLK_STONE;
                if (pn > 0.8f) blk = BLK_PATH;
                if (mv > 0.85f && wn > 0.4f) blk = BLK_PATH;
                if (mv > 0.85f && tn > 0.7f) blk = BLK_LAVA;
            }
            if (blk == BLK_STONE) {
                float ore = rand_f(&s.rng);
                if (ore < 0.005f && mv > 0.8f) blk = BLK_DIAMOND;
                else if (ore < 0.035f) blk = BLK_IRON;
                else if (ore < 0.075f) blk = BLK_COAL;
            }
            if (blk == BLK_GRASS && tn > 0.5f && rand_f(&s.rng) > 0.8f) blk = BLK_TREE;
            if (blk == BLK_DIAMOND) has_diamond = true;
            map_set(s, r, c, blk);
        }
    }
    map_set(s, center, center, BLK_GRASS);
    if (!has_diamond) {
        for (int a = 0; a < 1000; a++) {
            int r = rand_int(&s.rng, MAP_SIZE), c = rand_int(&s.rng, MAP_SIZE);
            if (map_get(s, r, c) == BLK_STONE) { map_set(s, r, c, BLK_DIAMOND); break; }
        }
    }
    s.player_r = center; s.player_c = center; s.player_dir = 4;
    s.health = 9; s.food = 9; s.drink = 9; s.energy = 9; s.is_sleeping = false;
    s.recover = 0; s.hunger = 0; s.thirst = 0; s.fatigue = 0;
    for (int i = 0; i < NUM_INVENTORY; i++) s.inv[i] = 0;
    for (int i = 0; i < MAX_ZOMBIES; i++) { s.zombie_mask[i] = false; s.zombie_hp[i] = 0; s.zombie_cd[i] = 0; }
    for (int i = 0; i < MAX_COWS; i++) { s.cow_mask[i] = false; s.cow_hp[i] = 0; }
    for (int i = 0; i < MAX_SKELETONS; i++) { s.skel_mask[i] = false; s.skel_hp[i] = 0; s.skel_cd[i] = 0; }
    for (int i = 0; i < MAX_ARROWS; i++) { s.arrow_mask[i] = false; }
    for (int i = 0; i < MAX_PLANTS; i++) { s.plant_mask[i] = false; s.plant_age[i] = 0; }
    s.achievements = 0u; s.timestep = 0; s.light_level = 1.0f;
}

__device__ void do_crafting(EnvState& s, int action) {
    if (action < ACT_MAKE_WOOD_PICK || action > ACT_MAKE_IRON_SWORD) return;
    bool near_table = is_near_block(s, BLK_TABLE);
    bool near_furnace = is_near_block(s, BLK_FURNACE);
    if (action == ACT_MAKE_WOOD_PICK && near_table && s.inv[0] >= 1) { s.inv[0]--; s.inv[6]++; ach_set(s, ACH_MAKE_WOOD_PICK); }
    if (action == ACT_MAKE_STONE_PICK && near_table && s.inv[0] >= 1 && s.inv[1] >= 1) { s.inv[0]--; s.inv[1]--; s.inv[7]++; ach_set(s, ACH_MAKE_STONE_PICK); }
    if (action == ACT_MAKE_IRON_PICK && near_table && near_furnace && s.inv[0] >= 1 && s.inv[1] >= 1 && s.inv[3] >= 1 && s.inv[2] >= 1) { s.inv[0]--; s.inv[1]--; s.inv[3]--; s.inv[2]--; s.inv[8]++; ach_set(s, ACH_MAKE_IRON_PICK); }
    if (action == ACT_MAKE_WOOD_SWORD && near_table && s.inv[0] >= 1) { s.inv[0]--; s.inv[9]++; ach_set(s, ACH_MAKE_WOOD_SWORD); }
    if (action == ACT_MAKE_STONE_SWORD && near_table && s.inv[0] >= 1 && s.inv[1] >= 1) { s.inv[0]--; s.inv[1]--; s.inv[10]++; ach_set(s, ACH_MAKE_STONE_SWORD); }
    if (action == ACT_MAKE_IRON_SWORD && near_table && near_furnace && s.inv[0] >= 1 && s.inv[1] >= 1 && s.inv[3] >= 1 && s.inv[2] >= 1) { s.inv[0]--; s.inv[1]--; s.inv[3]--; s.inv[2]--; s.inv[11]++; ach_set(s, ACH_MAKE_IRON_SWORD); }
}

__device__ void do_action(EnvState& s) {
    int tr = s.player_r + DIR_DR[s.player_dir];
    int tc = s.player_c + DIR_DC[s.player_dir];
    if (!in_bounds(tr, tc)) return;
    int dmg = get_damage(s);
    bool attacked = false;
    for (int i = 0; i < MAX_ZOMBIES && !attacked; i++) {
        if (s.zombie_mask[i] && s.zombie_r[i] == tr && s.zombie_c[i] == tc) {
            s.zombie_hp[i] -= dmg;
            if (s.zombie_hp[i] <= 0) { s.zombie_mask[i] = false; ach_set(s, ACH_DEFEAT_ZOMBIE); }
            attacked = true;
        }
    }
    for (int i = 0; i < MAX_COWS && !attacked; i++) {
        if (s.cow_mask[i] && s.cow_r[i] == tr && s.cow_c[i] == tc) {
            s.cow_hp[i] -= dmg;
            if (s.cow_hp[i] <= 0) { s.cow_mask[i] = false; ach_set(s, ACH_EAT_COW); s.food = min_i(9, s.food + 6); s.hunger = 0; }
            attacked = true;
        }
    }
    for (int i = 0; i < MAX_SKELETONS && !attacked; i++) {
        if (s.skel_mask[i] && s.skel_r[i] == tr && s.skel_c[i] == tc) {
            s.skel_hp[i] -= dmg;
            if (s.skel_hp[i] <= 0) { s.skel_mask[i] = false; ach_set(s, ACH_DEFEAT_SKELETON); }
            attacked = true;
        }
    }
    if (attacked) return;
    int8_t blk = map_get(s, tr, tc);
    switch (blk) {
        case BLK_TREE: map_set(s, tr, tc, BLK_GRASS); s.inv[0] = min_i(9, s.inv[0] + 1); ach_set(s, ACH_COLLECT_WOOD); break;
        case BLK_STONE: if (s.inv[6] > 0 || s.inv[7] > 0 || s.inv[8] > 0) { map_set(s, tr, tc, BLK_PATH); s.inv[1] = min_i(9, s.inv[1] + 1); ach_set(s, ACH_COLLECT_STONE); } break;
        case BLK_COAL: if (s.inv[6] > 0 || s.inv[7] > 0 || s.inv[8] > 0) { map_set(s, tr, tc, BLK_PATH); s.inv[2] = min_i(9, s.inv[2] + 1); ach_set(s, ACH_COLLECT_COAL); } break;
        case BLK_IRON: if (s.inv[7] > 0 || s.inv[8] > 0) { map_set(s, tr, tc, BLK_PATH); s.inv[3] = min_i(9, s.inv[3] + 1); ach_set(s, ACH_COLLECT_IRON); } break;
        case BLK_DIAMOND: if (s.inv[8] > 0) { map_set(s, tr, tc, BLK_PATH); s.inv[4] = min_i(9, s.inv[4] + 1); ach_set(s, ACH_COLLECT_DIAMOND); } break;
        case BLK_GRASS: if (rand_f(&s.rng) < 0.1f) { s.inv[5] = min_i(9, s.inv[5] + 1); ach_set(s, ACH_COLLECT_SAPLING); } break;
        case BLK_WATER: s.drink = min_i(9, s.drink + 1); s.thirst = 0; ach_set(s, ACH_COLLECT_DRINK); break;
        case BLK_RIPE_PLANT:
            map_set(s, tr, tc, BLK_PLANT); s.food = min_i(9, s.food + 4); s.hunger = 0; ach_set(s, ACH_EAT_PLANT);
            for (int i = 0; i < MAX_PLANTS; i++) if (s.plant_mask[i] && s.plant_r[i] == tr && s.plant_c[i] == tc) { s.plant_age[i] = 0; break; }
            break;
    }
}

__device__ void place_block(EnvState& s, int action) {
    int tr = s.player_r + DIR_DR[s.player_dir], tc = s.player_c + DIR_DC[s.player_dir];
    if (!in_bounds(tr, tc)) return;
    if (has_mob_at(s, tr, tc)) return;
    int8_t blk = map_get(s, tr, tc);
    if (action == ACT_PLACE_TABLE && s.inv[0] >= 2 && !is_solid(blk)) { map_set(s, tr, tc, BLK_TABLE); s.inv[0] -= 2; ach_set(s, ACH_PLACE_TABLE); }
    else if (action == ACT_PLACE_FURNACE && s.inv[1] >= 1 && !is_solid(blk)) { map_set(s, tr, tc, BLK_FURNACE); s.inv[1] -= 1; ach_set(s, ACH_PLACE_FURNACE); }
    else if (action == ACT_PLACE_STONE && s.inv[1] >= 1 && (!is_solid(blk) || blk == BLK_WATER)) { map_set(s, tr, tc, BLK_STONE); s.inv[1] -= 1; ach_set(s, ACH_PLACE_STONE); }
    else if (action == ACT_PLACE_PLANT && s.inv[5] >= 1 && blk == BLK_GRASS) {
        map_set(s, tr, tc, BLK_PLANT); s.inv[5] -= 1; ach_set(s, ACH_PLACE_PLANT);
        for (int i = 0; i < MAX_PLANTS; i++) if (!s.plant_mask[i]) { s.plant_r[i] = tr; s.plant_c[i] = tc; s.plant_age[i] = 0; s.plant_mask[i] = true; break; }
    }
}

__device__ void move_player(EnvState& s, int action) {
    if (action < 1 || action > 4) return;
    int nr = s.player_r + DIR_DR[action], nc = s.player_c + DIR_DC[action];
    s.player_dir = action;
    if (!in_bounds(nr, nc)) return;
    if (is_solid(map_get(s, nr, nc))) return;
    if (has_mob_at(s, nr, nc)) return;
    s.player_r = nr; s.player_c = nc;
}

__device__ bool can_move_mob(const EnvState& s, int r, int c) {
    if (!in_bounds(r, c)) return false;
    int8_t blk = map_get(s, r, c);
    if (is_solid(blk)) return false;
    if (blk == BLK_LAVA) return false;
    if (has_mob_at(s, r, c)) return false;
    if (r == s.player_r && c == s.player_c) return false;
    return true;
}

__device__ void update_mobs(EnvState& s) {
    int pr = s.player_r, pc = s.player_c;
    for (int i = 0; i < MAX_ZOMBIES; i++) {
        if (!s.zombie_mask[i]) continue;
        int zr = s.zombie_r[i], zc = s.zombie_c[i];
        int dist = l1_dist(zr, zc, pr, pc);
        if (dist >= MOB_DESPAWN_DIST) { s.zombie_mask[i] = false; continue; }
        if (dist <= 1 && s.zombie_cd[i] <= 0) { int dmg = s.is_sleeping ? 7 : 2; s.health -= dmg; s.zombie_cd[i] = 5; s.is_sleeping = false; }
        s.zombie_cd[i] = max_i(0, s.zombie_cd[i] - 1);
        int dr = 0, dc = 0;
        if (dist < 10 && rand_f(&s.rng) < 0.75f) {
            int adr = abs(pr - zr), adc = abs(pc - zc);
            if (adr > adc || (adr == adc && rand_f(&s.rng) < 0.5f)) dr = sign_i(pr - zr);
            else dc = sign_i(pc - zc);
        } else { int d = rand_int(&s.rng, 4); dr = DIR_DR[d+1]; dc = DIR_DC[d+1]; }
        int nr = zr + dr, nc = zc + dc;
        if (can_move_mob(s, nr, nc)) { s.zombie_r[i] = nr; s.zombie_c[i] = nc; }
    }
    for (int i = 0; i < MAX_COWS; i++) {
        if (!s.cow_mask[i]) continue;
        int cr = s.cow_r[i], cc = s.cow_c[i];
        int dist = l1_dist(cr, cc, pr, pc);
        if (dist >= MOB_DESPAWN_DIST) { s.cow_mask[i] = false; continue; }
        int d = rand_int(&s.rng, 8);
        if (d < 4) {
            int dr = DIR_DR[d+1], dc2 = DIR_DC[d+1];
            int nr = cr + dr, nc = cc + dc2;
            if (can_move_mob(s, nr, nc)) { s.cow_r[i] = nr; s.cow_c[i] = nc; }
        }
    }
    for (int i = 0; i < MAX_SKELETONS; i++) {
        if (!s.skel_mask[i]) continue;
        int sr = s.skel_r[i], sc = s.skel_c[i];
        int dist = l1_dist(sr, sc, pr, pc);
        if (dist >= MOB_DESPAWN_DIST) { s.skel_mask[i] = false; continue; }
        if (dist >= 4 && dist <= 5 && s.skel_cd[i] <= 0) {
            for (int a = 0; a < MAX_ARROWS; a++) {
                if (!s.arrow_mask[a]) {
                    s.arrow_mask[a] = true; s.arrow_r[a] = sr; s.arrow_c[a] = sc;
                    int adr = abs(pr - sr), adc = abs(pc - sc);
                    s.arrow_dr[a] = (adr > 0) ? sign_i(pr - sr) : 0;
                    s.arrow_dc[a] = (adc > 0) ? sign_i(pc - sc) : 0;
                    break;
                }
            }
            s.skel_cd[i] = 4;
        }
        s.skel_cd[i] = max_i(0, s.skel_cd[i] - 1);
        int dr = 0, dc = 0;
        bool rm = rand_f(&s.rng) < 0.15f;
        if (!rm) {
            if (dist >= 10) {
                int adr = abs(pr - sr), adc = abs(pc - sc);
                if (adr > adc || (adr == adc && rand_f(&s.rng) < 0.5f)) dr = sign_i(pr - sr);
                else dc = sign_i(pc - sc);
            } else if (dist <= 3) {
                int adr = abs(pr - sr), adc = abs(pc - sc);
                if (adr > adc || (adr == adc && rand_f(&s.rng) < 0.5f)) dr = -sign_i(pr - sr);
                else dc = -sign_i(pc - sc);
            } else rm = true;
        }
        if (rm) { int d = rand_int(&s.rng, 4); dr = DIR_DR[d+1]; dc = DIR_DC[d+1]; }
        int nr = sr + dr, nc = sc + dc;
        if (can_move_mob(s, nr, nc)) { s.skel_r[i] = nr; s.skel_c[i] = nc; }
    }
    for (int i = 0; i < MAX_ARROWS; i++) {
        if (!s.arrow_mask[i]) continue;
        int nr = s.arrow_r[i] + s.arrow_dr[i], nc = s.arrow_c[i] + s.arrow_dc[i];
        if (!in_bounds(nr, nc)) { s.arrow_mask[i] = false; continue; }
        int8_t blk = map_get(s, nr, nc);
        if (is_solid(blk) && blk != BLK_WATER) {
            if (blk == BLK_FURNACE || blk == BLK_TABLE) map_set(s, nr, nc, BLK_PATH);
            s.arrow_mask[i] = false; continue;
        }
        if (nr == pr && nc == pc) { s.health -= 2; s.is_sleeping = false; s.arrow_mask[i] = false; continue; }
        s.arrow_r[i] = nr; s.arrow_c[i] = nc;
    }
}

__device__ void spawn_mobs(EnvState& s) {
    int pr = s.player_r, pc = s.player_c;
    int n_cows = 0, n_z = 0, n_s = 0;
    for (int i = 0; i < MAX_COWS; i++) n_cows += s.cow_mask[i];
    for (int i = 0; i < MAX_ZOMBIES; i++) n_z += s.zombie_mask[i];
    for (int i = 0; i < MAX_SKELETONS; i++) n_s += s.skel_mask[i];
    auto try_spawn = [&](int mind, int maxd, bool ng, bool np, int* or_, int* oc) -> bool {
        for (int a = 0; a < 20; a++) {
            int r = rand_int(&s.rng, MAP_SIZE), c = rand_int(&s.rng, MAP_SIZE);
            int dist = l1_dist(r, c, pr, pc);
            if (dist < mind || dist >= maxd) continue;
            if (has_mob_at(s, r, c)) continue;
            if (r == pr && c == pc) continue;
            int8_t blk = map_get(s, r, c);
            if (ng && blk != BLK_GRASS) continue;
            if (np && blk != BLK_PATH) continue;
            if (!ng && !np && blk != BLK_GRASS && blk != BLK_PATH) continue;
            *or_ = r; *oc = c; return true;
        }
        return false;
    };
    if (n_cows < MAX_COWS && rand_f(&s.rng) < 0.1f) {
        int r, c;
        if (try_spawn(3, MOB_DESPAWN_DIST, true, false, &r, &c))
            for (int i = 0; i < MAX_COWS; i++) if (!s.cow_mask[i]) { s.cow_mask[i] = true; s.cow_r[i] = r; s.cow_c[i] = c; s.cow_hp[i] = 3; break; }
    }
    float zc = 0.02f + 0.1f * (1.0f - s.light_level) * (1.0f - s.light_level);
    if (n_z < MAX_ZOMBIES && rand_f(&s.rng) < zc) {
        int r, c;
        if (try_spawn(9, MOB_DESPAWN_DIST, false, false, &r, &c))
            for (int i = 0; i < MAX_ZOMBIES; i++) if (!s.zombie_mask[i]) { s.zombie_mask[i] = true; s.zombie_r[i] = r; s.zombie_c[i] = c; s.zombie_hp[i] = 5; s.zombie_cd[i] = 0; break; }
    }
    if (n_s < MAX_SKELETONS && rand_f(&s.rng) < 0.05f) {
        int r, c;
        if (try_spawn(9, MOB_DESPAWN_DIST, false, true, &r, &c))
            for (int i = 0; i < MAX_SKELETONS; i++) if (!s.skel_mask[i]) { s.skel_mask[i] = true; s.skel_r[i] = r; s.skel_c[i] = c; s.skel_hp[i] = 3; s.skel_cd[i] = 0; break; }
    }
}

__device__ void update_plants(EnvState& s) {
    for (int i = 0; i < MAX_PLANTS; i++) {
        if (!s.plant_mask[i]) continue;
        s.plant_age[i]++;
        if (s.plant_age[i] >= 600) {
            int r = s.plant_r[i], c = s.plant_c[i];
            if (in_bounds(r, c) && map_get(s, r, c) == BLK_PLANT) map_set(s, r, c, BLK_RIPE_PLANT);
        }
    }
}

__device__ void update_intrinsics(EnvState& s, int action) {
    if (action == ACT_SLEEP && s.energy < 9) s.is_sleeping = true;
    if (s.energy >= 9 && s.is_sleeping) { s.is_sleeping = false; ach_set(s, ACH_WAKE_UP); }
    float sm = s.is_sleeping ? 0.5f : 1.0f;
    s.hunger += sm; if (s.hunger > 25.0f) { s.food--; s.hunger = 0; }
    s.thirst += sm; if (s.thirst > 20.0f) { s.drink--; s.thirst = 0; }
    if (s.is_sleeping) s.fatigue -= 1.0f; else s.fatigue += 1.0f;
    if (s.fatigue > 30.0f) { s.energy--; s.fatigue = 0; }
    if (s.fatigue < -10.0f) { s.energy = min_i(s.energy + 1, 9); s.fatigue = 0; }
    bool all_n = (s.food > 0) && (s.drink > 0) && (s.energy > 0 || s.is_sleeping);
    if (all_n) s.recover += s.is_sleeping ? 2.0f : 1.0f;
    else s.recover += s.is_sleeping ? -0.5f : -1.0f;
    if (s.recover > 25.0f) { s.health = min_i(s.health + 1, 9); s.recover = 0; }
    if (s.recover < -15.0f) { s.health--; s.recover = 0; }
}

// === Warp-parallel observation builder ==================================
// 32 lanes of the warp cooperate on 1 env's obs. Works from the env's
// shared-memory EnvState `s`, writes directly to global `obs_out`.
//
// Layout: [63 cells × 21 channels] = 1323 floats, then 22 flat floats.
__device__ void build_observation_warp(const EnvState& s, float* obs_out, int lane) {
    constexpr int N_CELLS = OBS_MAP_ROWS * OBS_MAP_COLS;  // 63
    constexpr int CELL_STRIDE = OBS_MAP_CHANNELS;         // 21
    constexpr int N_MAP = N_CELLS * CELL_STRIDE;          // 1323

    // 1) Zero the full 1345-float obs buffer: coalesced across lanes.
    #pragma unroll 4
    for (int i = lane; i < OBS_DIM; i += 32) obs_out[i] = 0.0f;
    __syncwarp();

    // 2) Block one-hot: each lane handles cells [lane, lane+32) modulo stride.
    //    63 cells / 32 lanes -> 2 cells per lane (some get 1).
    int pr = s.player_r, pc = s.player_c;
    for (int cell = lane; cell < N_CELLS; cell += 32) {
        int dr = (cell / OBS_MAP_COLS) - 3;
        int dc = (cell % OBS_MAP_COLS) - 4;
        int r = pr + dr, c = pc + dc;
        int8_t blk = in_bounds(r, c) ? map_get(s, r, c) : BLK_OUT_OF_BOUNDS;
        obs_out[cell * CELL_STRIDE + blk] = 1.0f;
    }
    __syncwarp();

    // 3) Mob stamping: 11 mob slots, one per lane for lanes 0..10.
    //    Serialized writes per lane to avoid same-cell collisions.
    if (lane < MAX_ZOMBIES) {
        int i = lane;
        if (s.zombie_mask[i]) {
            int dr = s.zombie_r[i] - pr, dc = s.zombie_c[i] - pc;
            if (dr >= -3 && dr <= 3 && dc >= -4 && dc <= 4 && in_bounds(s.zombie_r[i], s.zombie_c[i])) {
                int cell = (dr + 3) * OBS_MAP_COLS + (dc + 4);
                obs_out[cell * CELL_STRIDE + NUM_BLOCK_TYPES + 0] = 1.0f;
            }
        }
    } else if (lane < MAX_ZOMBIES + MAX_COWS) {
        int i = lane - MAX_ZOMBIES;
        if (s.cow_mask[i]) {
            int dr = s.cow_r[i] - pr, dc = s.cow_c[i] - pc;
            if (dr >= -3 && dr <= 3 && dc >= -4 && dc <= 4 && in_bounds(s.cow_r[i], s.cow_c[i])) {
                int cell = (dr + 3) * OBS_MAP_COLS + (dc + 4);
                obs_out[cell * CELL_STRIDE + NUM_BLOCK_TYPES + 1] = 1.0f;
            }
        }
    } else if (lane < MAX_ZOMBIES + MAX_COWS + MAX_SKELETONS) {
        int i = lane - MAX_ZOMBIES - MAX_COWS;
        if (s.skel_mask[i]) {
            int dr = s.skel_r[i] - pr, dc = s.skel_c[i] - pc;
            if (dr >= -3 && dr <= 3 && dc >= -4 && dc <= 4 && in_bounds(s.skel_r[i], s.skel_c[i])) {
                int cell = (dr + 3) * OBS_MAP_COLS + (dc + 4);
                obs_out[cell * CELL_STRIDE + NUM_BLOCK_TYPES + 2] = 1.0f;
            }
        }
    } else if (lane < MAX_ZOMBIES + MAX_COWS + MAX_SKELETONS + MAX_ARROWS) {
        int i = lane - MAX_ZOMBIES - MAX_COWS - MAX_SKELETONS;
        if (s.arrow_mask[i]) {
            int dr = s.arrow_r[i] - pr, dc = s.arrow_c[i] - pc;
            if (dr >= -3 && dr <= 3 && dc >= -4 && dc <= 4 && in_bounds(s.arrow_r[i], s.arrow_c[i])) {
                int cell = (dr + 3) * OBS_MAP_COLS + (dc + 4);
                obs_out[cell * CELL_STRIDE + NUM_BLOCK_TYPES + 3] = 1.0f;
            }
        }
    }
    __syncwarp();

    // 4) Flat tail: 22 values. Lane `l` (l<22) writes one float.
    if (lane < NUM_INVENTORY) {
        obs_out[N_MAP + lane] = (float)s.inv[lane] / 10.0f;
    } else if (lane == NUM_INVENTORY + 0) obs_out[N_MAP + NUM_INVENTORY + 0] = (float)s.health / 10.0f;
    else if (lane == NUM_INVENTORY + 1) obs_out[N_MAP + NUM_INVENTORY + 1] = (float)s.food   / 10.0f;
    else if (lane == NUM_INVENTORY + 2) obs_out[N_MAP + NUM_INVENTORY + 2] = (float)s.drink  / 10.0f;
    else if (lane == NUM_INVENTORY + 3) obs_out[N_MAP + NUM_INVENTORY + 3] = (float)s.energy / 10.0f;
    else if (lane >= NUM_INVENTORY + 4 && lane < NUM_INVENTORY + 8) {
        int d = lane - (NUM_INVENTORY + 4) + 1;  // direction 1..4
        obs_out[N_MAP + NUM_INVENTORY + 4 + (d - 1)] = (s.player_dir == d) ? 1.0f : 0.0f;
    }
    else if (lane == NUM_INVENTORY + 8) obs_out[N_MAP + NUM_INVENTORY + 8] = s.light_level;
    else if (lane == NUM_INVENTORY + 9) obs_out[N_MAP + NUM_INVENTORY + 9] = s.is_sleeping ? 1.0f : 0.0f;
    __syncwarp();
}

// === Kernels =============================================================

extern "C" __global__ void reset_kernel(
    EnvState* __restrict__ states_g, float* __restrict__ obs_g,
    int num_envs, uint64_t seed
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int env_idx = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (env_idx >= num_envs) return;

    __shared__ EnvState s_envs[WARPS_PER_BLOCK];
    EnvState& s = s_envs[warp];

    // Lane 0 runs the serial worldgen into shared state.
    if (lane == 0) {
        generate_world(s, seed, env_idx);
    }
    __syncwarp();

    // Parallel obs build (direct to global).
    build_observation_warp(s, obs_g + env_idx * OBS_DIM, lane);

    // Warp-cooperative 128-bit store state back to global.
    warp_bulk_copy_i4(reinterpret_cast<int4*>(&states_g[env_idx]),
                      reinterpret_cast<int4*>(&s), lane);
}

extern "C" __global__ void step_only_kernel(
    EnvState* __restrict__ states_g, const int32_t* __restrict__ actions_g,
    float* __restrict__ rewards_g, int8_t* __restrict__ dones_g,
    int num_envs
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int env_idx = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (env_idx >= num_envs) return;

    __shared__ EnvState s_envs[WARPS_PER_BLOCK];
    EnvState& s = s_envs[warp];

    warp_bulk_copy_i4(reinterpret_cast<int4*>(&s),
                      reinterpret_cast<int4*>(&states_g[env_idx]), lane);
    __syncwarp();

    if (lane == 0) {
        int action = actions_g[env_idx];
        int old_health = s.health;
        uint32_t old_ach = s.achievements;
        if (s.is_sleeping) action = ACT_NOOP;

        do_crafting(s, action);
        if (action == ACT_DO) do_action(s);
        if (action >= ACT_PLACE_STONE && action <= ACT_PLACE_PLANT) place_block(s, action);
        move_player(s, action);
        update_mobs(s);
        spawn_mobs(s);
        update_plants(s);
        update_intrinsics(s, actions_g[env_idx]);

        uint32_t* p = reinterpret_cast<uint32_t*>(&s.inv[0]);
        p[0] = __vminu4(p[0], 0x09090909u);
        p[1] = __vminu4(p[1], 0x09090909u);
        p[2] = __vminu4(p[2], 0x09090909u);

        s.timestep++;
        float t_frac = fmodf((float)s.timestep / (float)DAY_LENGTH, 1.0f) + 0.3f;
        float cos_val = __cosf(3.14159265f * t_frac);
        s.light_level = 1.0f - fabsf(cos_val * cos_val * cos_val);

        uint32_t new_unlocks = s.achievements & ~old_ach;
        float ach_reward = (float)__popc(new_unlocks);
        float health_reward = (float)(s.health - old_health) * 0.1f;
        rewards_g[env_idx] = ach_reward + health_reward;

        bool done = (s.timestep >= MAX_TIMESTEPS) || (s.health <= 0);
        if (in_bounds(s.player_r, s.player_c) && map_get(s, s.player_r, s.player_c) == BLK_LAVA)
            done = true;
        dones_g[env_idx] = done ? 1 : 0;
    }
    __syncwarp();

    warp_bulk_copy_i4(reinterpret_cast<int4*>(&states_g[env_idx]),
                      reinterpret_cast<int4*>(&s), lane);
}

extern "C" __global__ void autoreset_obs_kernel(
    EnvState* __restrict__ states_g, const int8_t* __restrict__ dones_g,
    float* __restrict__ obs_g, int num_envs, uint64_t reset_seed
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int env_idx = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (env_idx >= num_envs) return;

    __shared__ EnvState s_envs[WARPS_PER_BLOCK];
    EnvState& s = s_envs[warp];

    warp_bulk_copy_i4(reinterpret_cast<int4*>(&s),
                      reinterpret_cast<int4*>(&states_g[env_idx]), lane);
    __syncwarp();

    int8_t d = dones_g[env_idx];
    if (d && lane == 0) generate_world(s, reset_seed, env_idx + num_envs);
    __syncwarp();

    build_observation_warp(s, obs_g + env_idx * OBS_DIM, lane);

    if (d) {
        warp_bulk_copy_i4(reinterpret_cast<int4*>(&states_g[env_idx]),
                          reinterpret_cast<int4*>(&s), lane);
    }
}

// ========================================================================
// MULTI-STEP fused kernel. Runs K game steps per launch, amortizing
// the warp-cooperative state load/store cost across K steps.
//
//   actions_ms[k, env_idx]   (K × NE int32)
//   rewards_ms[k, env_idx]   (K × NE float)
//   dones_ms[k, env_idx]     (K × NE int8)
//   obs_ms[k, env_idx, :]    (K × NE × OBS_DIM float)
//
// State is loaded once at entry, stays in shared across all K steps,
// written back once at exit.
// ========================================================================
extern "C" __global__ void step_fused_multistep_kernel(
    EnvState* __restrict__ states_g,
    const int32_t* __restrict__ actions_ms,
    float* __restrict__ rewards_ms,
    int8_t* __restrict__ dones_ms,
    float* __restrict__ obs_ms,
    int num_envs, int K, uint64_t reset_seed_base
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int env_idx = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (env_idx >= num_envs) return;

    __shared__ EnvState s_envs[WARPS_PER_BLOCK];
    EnvState& s = s_envs[warp];

    // Load once.
    warp_bulk_copy_i4(reinterpret_cast<int4*>(&s),
                      reinterpret_cast<int4*>(&states_g[env_idx]), lane);
    __syncwarp();

    for (int k = 0; k < K; k++) {
        if (lane == 0) {
            int action = actions_ms[k * num_envs + env_idx];
            int old_health = s.health;
            uint32_t old_ach = s.achievements;
            if (s.is_sleeping) action = ACT_NOOP;

            do_crafting(s, action);
            if (action == ACT_DO) do_action(s);
            if (action >= ACT_PLACE_STONE && action <= ACT_PLACE_PLANT) place_block(s, action);
            move_player(s, action);
            update_mobs(s);
            spawn_mobs(s);
            update_plants(s);
            update_intrinsics(s, actions_ms[k * num_envs + env_idx]);

            uint32_t* p = reinterpret_cast<uint32_t*>(&s.inv[0]);
            p[0] = __vminu4(p[0], 0x09090909u);
            p[1] = __vminu4(p[1], 0x09090909u);
            p[2] = __vminu4(p[2], 0x09090909u);

            s.timestep++;
            float t_frac = fmodf((float)s.timestep / (float)DAY_LENGTH, 1.0f) + 0.3f;
            float cos_val = __cosf(3.14159265f * t_frac);
            s.light_level = 1.0f - fabsf(cos_val * cos_val * cos_val);

            uint32_t new_unlocks = s.achievements & ~old_ach;
            float ach_reward = (float)__popc(new_unlocks);
            float health_reward = (float)(s.health - old_health) * 0.1f;
            rewards_ms[k * num_envs + env_idx] = ach_reward + health_reward;

            bool done = (s.timestep >= MAX_TIMESTEPS) || (s.health <= 0);
            if (in_bounds(s.player_r, s.player_c) && map_get(s, s.player_r, s.player_c) == BLK_LAVA)
                done = true;
            dones_ms[k * num_envs + env_idx] = done ? 1 : 0;

            // Autoreset for this step's output; seed must match single-step kernel:
            //   single kernel uses reset_seed = seed + step_count*1e6
            //   here we mimic by reset_seed_base + k*1e6 (caller sets base = seed + step_count*1e6)
            if (done) {
                uint64_t rs = reset_seed_base + (uint64_t)k * 1000000ULL;
                generate_world(s, rs, env_idx + num_envs);
            }
        }
        __syncwarp();

        // Obs build for step k.
        build_observation_warp(s, obs_ms + ((size_t)k * num_envs + env_idx) * OBS_DIM, lane);
    }

    // Store once.
    warp_bulk_copy_i4(reinterpret_cast<int4*>(&states_g[env_idx]),
                      reinterpret_cast<int4*>(&s), lane);
}

// ========================================================================
// FUSED step + autoreset + obs kernel.
// Single kernel, single state load, single state store. Amortizes the
// warp-cooperative bulk copies across all three old kernel phases.
// ========================================================================
extern "C" __global__ void step_fused_kernel(
    EnvState* __restrict__ states_g, const int32_t* __restrict__ actions_g,
    float* __restrict__ rewards_g, int8_t* __restrict__ dones_g,
    float* __restrict__ obs_g, int num_envs, uint64_t reset_seed
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int env_idx = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (env_idx >= num_envs) return;

    __shared__ EnvState s_envs[WARPS_PER_BLOCK];
    __shared__ int map_dirty[WARPS_PER_BLOCK];
    EnvState& s = s_envs[warp];

    // Phase 1: warp bulk load (int4).
    warp_bulk_copy_i4(reinterpret_cast<int4*>(&s),
                      reinterpret_cast<int4*>(&states_g[env_idx]), lane);
    __syncwarp();

    // Phase 2: game step (lane 0).
    if (lane == 0) {
        int action = actions_g[env_idx];
        int old_health = s.health;
        uint32_t old_ach = s.achievements;
        if (s.is_sleeping) action = ACT_NOOP;

        // Pre-compute conservative map-dirty predicate before mutating state.
        // Dirty if: ACT_DO (can mine/harvest), PLACE_* (places), done (regen),
        //          any arrow alive (may hit furnace/table), or any plant about
        //          to ripen (age >= 599 this tick).
        bool had_arrow = false;
        for (int i = 0; i < MAX_ARROWS; i++) had_arrow |= s.arrow_mask[i];
        bool had_ripening = false;
        for (int i = 0; i < MAX_PLANTS; i++)
            if (s.plant_mask[i] && s.plant_age[i] >= 599) { had_ripening = true; break; }

        do_crafting(s, action);
        if (action == ACT_DO) do_action(s);
        if (action >= ACT_PLACE_STONE && action <= ACT_PLACE_PLANT) place_block(s, action);
        move_player(s, action);
        update_mobs(s);
        spawn_mobs(s);
        update_plants(s);
        update_intrinsics(s, actions_g[env_idx]);

        uint32_t* p = reinterpret_cast<uint32_t*>(&s.inv[0]);
        p[0] = __vminu4(p[0], 0x09090909u);
        p[1] = __vminu4(p[1], 0x09090909u);
        p[2] = __vminu4(p[2], 0x09090909u);

        s.timestep++;
        float t_frac = fmodf((float)s.timestep / (float)DAY_LENGTH, 1.0f) + 0.3f;
        float cos_val = __cosf(3.14159265f * t_frac);
        s.light_level = 1.0f - fabsf(cos_val * cos_val * cos_val);

        uint32_t new_unlocks = s.achievements & ~old_ach;
        float ach_reward = (float)__popc(new_unlocks);
        float health_reward = (float)(s.health - old_health) * 0.1f;
        rewards_g[env_idx] = ach_reward + health_reward;

        bool done = (s.timestep >= MAX_TIMESTEPS) || (s.health <= 0);
        if (in_bounds(s.player_r, s.player_c) && map_get(s, s.player_r, s.player_c) == BLK_LAVA)
            done = true;
        dones_g[env_idx] = done ? 1 : 0;

        // Phase 3: autoreset if done (still lane 0, still on shared).
        if (done) generate_world(s, reset_seed, env_idx + num_envs);

        bool dirty = (action == ACT_DO)
                  || (action >= ACT_PLACE_STONE && action <= ACT_PLACE_PLANT)
                  || done || had_arrow || had_ripening;
        map_dirty[warp] = dirty ? 1 : 0;
    }
    __syncwarp();

    // Phase 4: obs build (warp-parallel from shared).
    build_observation_warp(s, obs_g + env_idx * OBS_DIM, lane);

    // Phase 5: bulk store state back. Skip the 2 KB map region when clean.
    if (map_dirty[warp]) {
        warp_bulk_copy_i4(reinterpret_cast<int4*>(&states_g[env_idx]),
                          reinterpret_cast<int4*>(&s), lane);
    } else {
        warp_bulk_copy_nonmap_i4(reinterpret_cast<int4*>(&states_g[env_idx]),
                                 reinterpret_cast<int4*>(&s), lane);
    }
}
