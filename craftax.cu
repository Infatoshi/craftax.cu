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
// ============================================================
__device__ void generate_world(EnvState& s, uint64_t seed, int env_id) {
    curand_init(seed, env_id, 0, &s.rng);

    // Clear packed map to all GRASS
    for (int i = 0; i < MAP_SIZE * MAP_PACKED_ROW; i++)
        s.map_packed[i] = (BLK_GRASS | (BLK_GRASS << 4));

    const int GRID = 10;
    float angles[4][GRID * GRID];
    for (int layer = 0; layer < 4; layer++)
        for (int i = 0; i < GRID * GRID; i++)
            angles[layer][i] = rand_f(&s.rng) * 2.0f * 3.14159265f;

    float scale = (float)MAP_SIZE / (float)(GRID - 1);

    int center = MAP_SIZE / 2;
    for (int r = 0; r < MAP_SIZE; r++) {
        for (int c = 0; c < MAP_SIZE; c++) {
            float nr = (float)r / scale;
            float nc = (float)c / scale;

            float water_noise = perlin_2d(nr, nc, angles[0], GRID);
            float mountain_noise = perlin_2d(nr, nc, angles[1], GRID);
            float tree_noise = perlin_2d(nr, nc, angles[2], GRID);
            float path_noise = perlin_2d(nr, nc, angles[3], GRID);

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

            if (blk == BLK_STONE) {
                float ore_roll = rand_f(&s.rng);
                if (ore_roll < 0.005f && mountain_val > 0.8f)
                    blk = BLK_DIAMOND;
                else if (ore_roll < 0.035f)
                    blk = BLK_IRON;
                else if (ore_roll < 0.075f)
                    blk = BLK_COAL;
            }

            if (blk == BLK_GRASS && tree_noise > 0.5f && rand_f(&s.rng) > 0.8f)
                blk = BLK_TREE;

            map_set(s, r, c, blk);
        }
    }

    map_set(s, center, center, BLK_GRASS);

    bool has_diamond = false;
    for (int r = 0; r < MAP_SIZE && !has_diamond; r++)
        for (int c = 0; c < MAP_SIZE && !has_diamond; c++)
            if (map_get(s, r, c) == BLK_DIAMOND) has_diamond = true;
    if (!has_diamond) {
        for (int attempts = 0; attempts < 1000; attempts++) {
            int r = rand_int(&s.rng, MAP_SIZE);
            int c = rand_int(&s.rng, MAP_SIZE);
            if (map_get(s, r, c) == BLK_STONE) {
                map_set(s, r, c, BLK_DIAMOND);
                break;
            }
        }
    }

    s.player_r = center; s.player_c = center;
    s.player_dir = 4;
    s.health = 9; s.food = 9; s.drink = 9; s.energy = 9;
    s.is_sleeping = false;
    s.recover = 0; s.hunger = 0; s.thirst = 0; s.fatigue = 0;

    for (int i = 0; i < NUM_INVENTORY; i++) s.inv[i] = 0;

    for (int i = 0; i < MAX_ZOMBIES; i++) { s.zombie_mask[i] = false; s.zombie_hp[i] = 0; s.zombie_cd[i] = 0; }
    for (int i = 0; i < MAX_COWS; i++) { s.cow_mask[i] = false; s.cow_hp[i] = 0; }
    for (int i = 0; i < MAX_SKELETONS; i++) { s.skel_mask[i] = false; s.skel_hp[i] = 0; s.skel_cd[i] = 0; }
    for (int i = 0; i < MAX_ARROWS; i++) { s.arrow_mask[i] = false; }
    for (int i = 0; i < MAX_PLANTS; i++) { s.plant_mask[i] = false; s.plant_age[i] = 0; }

    for (int i = 0; i < NUM_ACHIEVEMENTS; i++) s.achievements[i] = false;

    s.timestep = 0;
    s.light_level = 1.0f;
}

// ============================================================
// Step Logic
// ============================================================
__device__ void do_crafting(EnvState& s, int action) {
    bool near_table = is_near_block(s, BLK_TABLE);
    bool near_furnace = is_near_block(s, BLK_FURNACE);

    if (action == ACT_MAKE_WOOD_PICK && near_table && s.inv[0] >= 1) {
        s.inv[0]--; s.inv[6]++; s.achievements[ACH_MAKE_WOOD_PICK] = true;
    }
    if (action == ACT_MAKE_STONE_PICK && near_table && s.inv[0] >= 1 && s.inv[1] >= 1) {
        s.inv[0]--; s.inv[1]--; s.inv[7]++; s.achievements[ACH_MAKE_STONE_PICK] = true;
    }
    if (action == ACT_MAKE_IRON_PICK && near_table && near_furnace &&
        s.inv[0] >= 1 && s.inv[1] >= 1 && s.inv[3] >= 1 && s.inv[2] >= 1) {
        s.inv[0]--; s.inv[1]--; s.inv[3]--; s.inv[2]--;
        s.inv[8]++; s.achievements[ACH_MAKE_IRON_PICK] = true;
    }
    if (action == ACT_MAKE_WOOD_SWORD && near_table && s.inv[0] >= 1) {
        s.inv[0]--; s.inv[9]++; s.achievements[ACH_MAKE_WOOD_SWORD] = true;
    }
    if (action == ACT_MAKE_STONE_SWORD && near_table && s.inv[0] >= 1 && s.inv[1] >= 1) {
        s.inv[0]--; s.inv[1]--; s.inv[10]++; s.achievements[ACH_MAKE_STONE_SWORD] = true;
    }
    if (action == ACT_MAKE_IRON_SWORD && near_table && near_furnace &&
        s.inv[0] >= 1 && s.inv[1] >= 1 && s.inv[3] >= 1 && s.inv[2] >= 1) {
        s.inv[0]--; s.inv[1]--; s.inv[3]--; s.inv[2]--;
        s.inv[11]++; s.achievements[ACH_MAKE_IRON_SWORD] = true;
    }
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
            if (s.zombie_hp[i] <= 0) {
                s.zombie_mask[i] = false;
                s.achievements[ACH_DEFEAT_ZOMBIE] = true;
            }
            attacked = true;
        }
    }
    for (int i = 0; i < MAX_COWS && !attacked; i++) {
        if (s.cow_mask[i] && s.cow_r[i] == tr && s.cow_c[i] == tc) {
            s.cow_hp[i] -= dmg;
            if (s.cow_hp[i] <= 0) {
                s.cow_mask[i] = false;
                s.achievements[ACH_EAT_COW] = true;
                s.food = min_i(9, s.food + 6);
                s.hunger = 0;
            }
            attacked = true;
        }
    }
    for (int i = 0; i < MAX_SKELETONS && !attacked; i++) {
        if (s.skel_mask[i] && s.skel_r[i] == tr && s.skel_c[i] == tc) {
            s.skel_hp[i] -= dmg;
            if (s.skel_hp[i] <= 0) {
                s.skel_mask[i] = false;
                s.achievements[ACH_DEFEAT_SKELETON] = true;
            }
            attacked = true;
        }
    }

    if (attacked) return;

    int8_t blk = map_get(s, tr, tc);
    switch (blk) {
        case BLK_TREE:
            map_set(s, tr, tc, BLK_GRASS);
            s.inv[0] = min_i(9, s.inv[0] + 1);
            s.achievements[ACH_COLLECT_WOOD] = true;
            break;
        case BLK_STONE:
            if (s.inv[6] > 0 || s.inv[7] > 0 || s.inv[8] > 0) {
                map_set(s, tr, tc, BLK_PATH);
                s.inv[1] = min_i(9, s.inv[1] + 1);
                s.achievements[ACH_COLLECT_STONE] = true;
            }
            break;
        case BLK_COAL:
            if (s.inv[6] > 0 || s.inv[7] > 0 || s.inv[8] > 0) {
                map_set(s, tr, tc, BLK_PATH);
                s.inv[2] = min_i(9, s.inv[2] + 1);
                s.achievements[ACH_COLLECT_COAL] = true;
            }
            break;
        case BLK_IRON:
            if (s.inv[7] > 0 || s.inv[8] > 0) {
                map_set(s, tr, tc, BLK_PATH);
                s.inv[3] = min_i(9, s.inv[3] + 1);
                s.achievements[ACH_COLLECT_IRON] = true;
            }
            break;
        case BLK_DIAMOND:
            if (s.inv[8] > 0) {
                map_set(s, tr, tc, BLK_PATH);
                s.inv[4] = min_i(9, s.inv[4] + 1);
                s.achievements[ACH_COLLECT_DIAMOND] = true;
            }
            break;
        case BLK_GRASS:
            if (rand_f(&s.rng) < 0.1f) {
                s.inv[5] = min_i(9, s.inv[5] + 1);
                s.achievements[ACH_COLLECT_SAPLING] = true;
            }
            break;
        case BLK_WATER:
            s.drink = min_i(9, s.drink + 1);
            s.thirst = 0;
            s.achievements[ACH_COLLECT_DRINK] = true;
            break;
        case BLK_RIPE_PLANT:
            map_set(s, tr, tc, BLK_PLANT);
            s.food = min_i(9, s.food + 4);
            s.hunger = 0;
            s.achievements[ACH_EAT_PLANT] = true;
            for (int i = 0; i < MAX_PLANTS; i++) {
                if (s.plant_mask[i] && s.plant_r[i] == tr && s.plant_c[i] == tc) {
                    s.plant_age[i] = 0;
                    break;
                }
            }
            break;
    }
}

__device__ void place_block(EnvState& s, int action) {
    int tr = s.player_r + DIR_DR[s.player_dir];
    int tc = s.player_c + DIR_DC[s.player_dir];
    if (!in_bounds(tr, tc)) return;
    if (has_mob_at(s, tr, tc)) return;

    int8_t blk = map_get(s, tr, tc);

    if (action == ACT_PLACE_TABLE && s.inv[0] >= 2 && !is_solid(blk)) {
        map_set(s, tr, tc, BLK_TABLE); s.inv[0] -= 2;
        s.achievements[ACH_PLACE_TABLE] = true;
    }
    else if (action == ACT_PLACE_FURNACE && s.inv[1] >= 1 && !is_solid(blk)) {
        map_set(s, tr, tc, BLK_FURNACE); s.inv[1] -= 1;
        s.achievements[ACH_PLACE_FURNACE] = true;
    }
    else if (action == ACT_PLACE_STONE && s.inv[1] >= 1 && (!is_solid(blk) || blk == BLK_WATER)) {
        map_set(s, tr, tc, BLK_STONE); s.inv[1] -= 1;
        s.achievements[ACH_PLACE_STONE] = true;
    }
    else if (action == ACT_PLACE_PLANT && s.inv[5] >= 1 && blk == BLK_GRASS) {
        map_set(s, tr, tc, BLK_PLANT); s.inv[5] -= 1;
        s.achievements[ACH_PLACE_PLANT] = true;
        for (int i = 0; i < MAX_PLANTS; i++) {
            if (!s.plant_mask[i]) {
                s.plant_r[i] = tr; s.plant_c[i] = tc;
                s.plant_age[i] = 0; s.plant_mask[i] = true;
                break;
            }
        }
    }
}

__device__ void move_player(EnvState& s, int action) {
    if (action < 1 || action > 4) return;
    int nr = s.player_r + DIR_DR[action];
    int nc = s.player_c + DIR_DC[action];
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

        if (dist <= 1 && s.zombie_cd[i] <= 0) {
            int dmg = s.is_sleeping ? 7 : 2;
            s.health -= dmg;
            s.zombie_cd[i] = 5;
            s.is_sleeping = false;
        }
        s.zombie_cd[i] = max_i(0, s.zombie_cd[i] - 1);

        int dr = 0, dc = 0;
        if (dist < 10 && rand_f(&s.rng) < 0.75f) {
            int adr = abs(pr - zr), adc = abs(pc - zc);
            if (adr > adc || (adr == adc && rand_f(&s.rng) < 0.5f))
                dr = sign_i(pr - zr);
            else
                dc = sign_i(pc - zc);
        } else {
            int d = rand_int(&s.rng, 4);
            dr = DIR_DR[d+1]; dc = DIR_DC[d+1];
        }
        int nr = zr + dr, nc = zc + dc;
        if (can_move_mob(s, nr, nc)) {
            s.zombie_r[i] = nr; s.zombie_c[i] = nc;
        }
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
            if (can_move_mob(s, nr, nc)) {
                s.cow_r[i] = nr; s.cow_c[i] = nc;
            }
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
                    s.arrow_mask[a] = true;
                    s.arrow_r[a] = sr; s.arrow_c[a] = sc;
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
        bool random_move = rand_f(&s.rng) < 0.15f;
        if (!random_move) {
            if (dist >= 10) {
                int adr = abs(pr - sr), adc = abs(pc - sc);
                if (adr > adc || (adr == adc && rand_f(&s.rng) < 0.5f))
                    dr = sign_i(pr - sr);
                else
                    dc = sign_i(pc - sc);
            } else if (dist <= 3) {
                int adr = abs(pr - sr), adc = abs(pc - sc);
                if (adr > adc || (adr == adc && rand_f(&s.rng) < 0.5f))
                    dr = -sign_i(pr - sr);
                else
                    dc = -sign_i(pc - sc);
            } else {
                random_move = true;
            }
        }
        if (random_move) {
            int d = rand_int(&s.rng, 4);
            dr = DIR_DR[d+1]; dc = DIR_DC[d+1];
        }
        int nr = sr + dr, nc = sc + dc;
        if (can_move_mob(s, nr, nc)) {
            s.skel_r[i] = nr; s.skel_c[i] = nc;
        }
    }

    for (int i = 0; i < MAX_ARROWS; i++) {
        if (!s.arrow_mask[i]) continue;
        int nr = s.arrow_r[i] + s.arrow_dr[i];
        int nc = s.arrow_c[i] + s.arrow_dc[i];

        if (!in_bounds(nr, nc)) { s.arrow_mask[i] = false; continue; }
        int8_t blk = map_get(s, nr, nc);
        if (is_solid(blk) && blk != BLK_WATER) {
            if (blk == BLK_FURNACE || blk == BLK_TABLE) map_set(s, nr, nc, BLK_PATH);
            s.arrow_mask[i] = false;
            continue;
        }
        if (nr == pr && nc == pc) {
            s.health -= 2;
            s.is_sleeping = false;
            s.arrow_mask[i] = false;
            continue;
        }
        s.arrow_r[i] = nr; s.arrow_c[i] = nc;
    }
}

__device__ void spawn_mobs(EnvState& s) {
    int pr = s.player_r, pc = s.player_c;

    int n_cows = 0, n_zombies = 0, n_skels = 0;
    for (int i = 0; i < MAX_COWS; i++) n_cows += s.cow_mask[i];
    for (int i = 0; i < MAX_ZOMBIES; i++) n_zombies += s.zombie_mask[i];
    for (int i = 0; i < MAX_SKELETONS; i++) n_skels += s.skel_mask[i];

    auto try_spawn = [&](int min_dist, int max_dist, bool need_grass, bool need_path, int* out_r, int* out_c) -> bool {
        for (int attempts = 0; attempts < 20; attempts++) {
            int r = rand_int(&s.rng, MAP_SIZE);
            int c = rand_int(&s.rng, MAP_SIZE);
            int dist = l1_dist(r, c, pr, pc);
            if (dist < min_dist || dist >= max_dist) continue;
            if (has_mob_at(s, r, c)) continue;
            if (r == pr && c == pc) continue;
            int8_t blk = map_get(s, r, c);
            if (need_grass && blk != BLK_GRASS) continue;
            if (need_path && blk != BLK_PATH) continue;
            if (!need_grass && !need_path && blk != BLK_GRASS && blk != BLK_PATH) continue;
            *out_r = r; *out_c = c;
            return true;
        }
        return false;
    };

    if (n_cows < MAX_COWS && rand_f(&s.rng) < 0.1f) {
        int r, c;
        if (try_spawn(3, MOB_DESPAWN_DIST, true, false, &r, &c)) {
            for (int i = 0; i < MAX_COWS; i++) {
                if (!s.cow_mask[i]) {
                    s.cow_mask[i] = true; s.cow_r[i] = r; s.cow_c[i] = c; s.cow_hp[i] = 3;
                    break;
                }
            }
        }
    }

    float zombie_chance = 0.02f + 0.1f * (1.0f - s.light_level) * (1.0f - s.light_level);
    if (n_zombies < MAX_ZOMBIES && rand_f(&s.rng) < zombie_chance) {
        int r, c;
        if (try_spawn(9, MOB_DESPAWN_DIST, false, false, &r, &c)) {
            for (int i = 0; i < MAX_ZOMBIES; i++) {
                if (!s.zombie_mask[i]) {
                    s.zombie_mask[i] = true; s.zombie_r[i] = r; s.zombie_c[i] = c;
                    s.zombie_hp[i] = 5; s.zombie_cd[i] = 0;
                    break;
                }
            }
        }
    }

    if (n_skels < MAX_SKELETONS && rand_f(&s.rng) < 0.05f) {
        int r, c;
        if (try_spawn(9, MOB_DESPAWN_DIST, false, true, &r, &c)) {
            for (int i = 0; i < MAX_SKELETONS; i++) {
                if (!s.skel_mask[i]) {
                    s.skel_mask[i] = true; s.skel_r[i] = r; s.skel_c[i] = c;
                    s.skel_hp[i] = 3; s.skel_cd[i] = 0;
                    break;
                }
            }
        }
    }
}

__device__ void update_plants(EnvState& s) {
    for (int i = 0; i < MAX_PLANTS; i++) {
        if (!s.plant_mask[i]) continue;
        s.plant_age[i]++;
        if (s.plant_age[i] >= 600) {
            int r = s.plant_r[i], c = s.plant_c[i];
            if (in_bounds(r, c) && map_get(s, r, c) == BLK_PLANT) {
                map_set(s, r, c, BLK_RIPE_PLANT);
            }
        }
    }
}

__device__ void update_intrinsics(EnvState& s, int action) {
    if (action == ACT_SLEEP && s.energy < 9) s.is_sleeping = true;
    if (s.energy >= 9 && s.is_sleeping) {
        s.is_sleeping = false;
        s.achievements[ACH_WAKE_UP] = true;
    }

    float sleep_mul = s.is_sleeping ? 0.5f : 1.0f;

    s.hunger += sleep_mul;
    if (s.hunger > 25.0f) { s.food--; s.hunger = 0; }

    s.thirst += sleep_mul;
    if (s.thirst > 20.0f) { s.drink--; s.thirst = 0; }

    if (s.is_sleeping) s.fatigue -= 1.0f;
    else s.fatigue += 1.0f;
    if (s.fatigue > 30.0f) { s.energy--; s.fatigue = 0; }
    if (s.fatigue < -10.0f) { s.energy = min_i(s.energy + 1, 9); s.fatigue = 0; }

    bool all_needs = (s.food > 0) && (s.drink > 0) && (s.energy > 0 || s.is_sleeping);
    if (all_needs) s.recover += s.is_sleeping ? 2.0f : 1.0f;
    else s.recover += s.is_sleeping ? -0.5f : -1.0f;
    if (s.recover > 25.0f) { s.health = min_i(s.health + 1, 9); s.recover = 0; }
    if (s.recover < -15.0f) { s.health--; s.recover = 0; }
}

__device__ void build_observation(const EnvState& s, float* obs) {
    int pr = s.player_r, pc = s.player_c;
    int obs_idx = 0;

    for (int dr = -3; dr <= 3; dr++) {
        for (int dc = -4; dc <= 4; dc++) {
            int r = pr + dr, c = pc + dc;
            int8_t blk = (in_bounds(r, c)) ? map_get(s, r, c) : BLK_OUT_OF_BOUNDS;

            for (int b = 0; b < NUM_BLOCK_TYPES; b++)
                obs[obs_idx++] = (blk == b) ? 1.0f : 0.0f;

            float mob_z = 0, mob_c = 0, mob_s = 0, mob_a = 0;
            if (in_bounds(r, c)) {
                for (int i = 0; i < MAX_ZOMBIES; i++)
                    if (s.zombie_mask[i] && s.zombie_r[i] == r && s.zombie_c[i] == c) mob_z = 1.0f;
                for (int i = 0; i < MAX_COWS; i++)
                    if (s.cow_mask[i] && s.cow_r[i] == r && s.cow_c[i] == c) mob_c = 1.0f;
                for (int i = 0; i < MAX_SKELETONS; i++)
                    if (s.skel_mask[i] && s.skel_r[i] == r && s.skel_c[i] == c) mob_s = 1.0f;
                for (int i = 0; i < MAX_ARROWS; i++)
                    if (s.arrow_mask[i] && s.arrow_r[i] == r && s.arrow_c[i] == c) mob_a = 1.0f;
            }
            obs[obs_idx++] = mob_z;
            obs[obs_idx++] = mob_c;
            obs[obs_idx++] = mob_s;
            obs[obs_idx++] = mob_a;
        }
    }

    for (int i = 0; i < NUM_INVENTORY; i++)
        obs[obs_idx++] = (float)s.inv[i] / 10.0f;

    obs[obs_idx++] = (float)s.health / 10.0f;
    obs[obs_idx++] = (float)s.food / 10.0f;
    obs[obs_idx++] = (float)s.drink / 10.0f;
    obs[obs_idx++] = (float)s.energy / 10.0f;

    for (int d = 1; d <= 4; d++)
        obs[obs_idx++] = (s.player_dir == d) ? 1.0f : 0.0f;

    obs[obs_idx++] = s.light_level;
    obs[obs_idx++] = s.is_sleeping ? 1.0f : 0.0f;
}

// ============================================================
// Main Kernels
// ============================================================
extern "C" __global__ void reset_kernel(EnvState* states, float* obs, int num_envs, uint64_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;

    generate_world(states[idx], seed, idx);
    build_observation(states[idx], obs + idx * OBS_DIM);
}

// Step kernel: game logic only, marks dones but does NOT auto-reset
extern "C" __global__ void step_only_kernel(
    EnvState* states, const int32_t* actions,
    float* rewards, int8_t* dones,
    int num_envs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;

    EnvState& s = states[idx];
    int action = actions[idx];

    int old_health = s.health;
    bool old_ach[NUM_ACHIEVEMENTS];
    for (int i = 0; i < NUM_ACHIEVEMENTS; i++) old_ach[i] = s.achievements[i];

    if (s.is_sleeping) action = ACT_NOOP;

    do_crafting(s, action);
    if (action == ACT_DO) do_action(s);
    if (action >= ACT_PLACE_STONE && action <= ACT_PLACE_PLANT) place_block(s, action);
    move_player(s, action);
    update_mobs(s);
    spawn_mobs(s);
    update_plants(s);
    update_intrinsics(s, actions[idx]);

    for (int i = 0; i < NUM_INVENTORY; i++)
        s.inv[i] = clamp_i(s.inv[i], 0, 9);

    s.timestep++;
    float t_frac = fmodf((float)s.timestep / (float)DAY_LENGTH, 1.0f) + 0.3f;
    float cos_val = __cosf(3.14159265f * t_frac);
    s.light_level = 1.0f - fabsf(cos_val * cos_val * cos_val);

    float ach_reward = 0;
    for (int i = 0; i < NUM_ACHIEVEMENTS; i++)
        ach_reward += (float)(s.achievements[i] && !old_ach[i]);
    float health_reward = (float)(s.health - old_health) * 0.1f;
    rewards[idx] = ach_reward + health_reward;

    bool done = (s.timestep >= MAX_TIMESTEPS) || (s.health <= 0);
    if (in_bounds(s.player_r, s.player_c) && map_get(s, s.player_r, s.player_c) == BLK_LAVA)
        done = true;
    dones[idx] = done ? 1 : 0;
}

// Auto-reset kernel: only runs on done envs, then builds obs for ALL envs
extern "C" __global__ void autoreset_obs_kernel(
    EnvState* states, const int8_t* dones,
    float* obs, int num_envs, uint64_t reset_seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;

    if (dones[idx]) {
        generate_world(states[idx], reset_seed, idx + num_envs);
    }
    build_observation(states[idx], obs + idx * OBS_DIM);
}

// Combined step kernel (for backward compat)
extern "C" __global__ void step_kernel(
    EnvState* states, const int32_t* actions,
    float* obs, float* rewards, int8_t* dones,
    int num_envs, uint64_t reset_seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;

    EnvState& s = states[idx];
    int action = actions[idx];

    int old_health = s.health;
    bool old_ach[NUM_ACHIEVEMENTS];
    for (int i = 0; i < NUM_ACHIEVEMENTS; i++) old_ach[i] = s.achievements[i];

    if (s.is_sleeping) action = ACT_NOOP;

    do_crafting(s, action);
    if (action == ACT_DO) do_action(s);
    if (action >= ACT_PLACE_STONE && action <= ACT_PLACE_PLANT) place_block(s, action);
    move_player(s, action);
    update_mobs(s);
    spawn_mobs(s);
    update_plants(s);
    update_intrinsics(s, actions[idx]);

    for (int i = 0; i < NUM_INVENTORY; i++)
        s.inv[i] = clamp_i(s.inv[i], 0, 9);

    s.timestep++;
    float t_frac = fmodf((float)s.timestep / (float)DAY_LENGTH, 1.0f) + 0.3f;
    float cos_val = __cosf(3.14159265f * t_frac);
    s.light_level = 1.0f - fabsf(cos_val * cos_val * cos_val);

    float ach_reward = 0;
    for (int i = 0; i < NUM_ACHIEVEMENTS; i++)
        ach_reward += (float)(s.achievements[i] && !old_ach[i]);
    float health_reward = (float)(s.health - old_health) * 0.1f;
    rewards[idx] = ach_reward + health_reward;

    bool done = (s.timestep >= MAX_TIMESTEPS) || (s.health <= 0);
    if (in_bounds(s.player_r, s.player_c) && map_get(s, s.player_r, s.player_c) == BLK_LAVA)
        done = true;
    dones[idx] = done ? 1 : 0;

    if (done) {
        generate_world(s, reset_seed, idx + num_envs);
    }
    build_observation(s, obs + idx * OBS_DIM);
}
