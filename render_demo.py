# Render CfFrame records from craftax_full.cu play mode into PNG frames.
# Usage: uv run --with numpy,pillow python render_demo.py frames.bin outdir
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

MAP = 48
TILE = 16
TEX = Path("/home/infatoshi/kernels/PufferLib/resources/craftax/textures.bin")

frame_dtype = np.dtype([
    ("map", np.uint8, (MAP, MAP)),
    ("item", np.uint8, (MAP, MAP)),
    ("light_q", np.uint8, (MAP, MAP)),
    ("mob", np.uint8, (15, 4)),
    ("level", np.int32), ("posr", np.int32), ("posc", np.int32),
    ("dir", np.int32), ("action", np.int32), ("done", np.int32),
    ("health", np.float32), ("reward", np.float32),
    ("light_level", np.float32),
    ("food", np.int32), ("drink", np.int32), ("energy", np.int32),
    ("mana", np.int32),
    ("inv", np.int32, 14), ("armour", np.int32, 4),
    ("potions", np.int32, 6), ("ach_count", np.int32),
])

tex = np.fromfile(TEX, dtype=np.uint8).reshape(-1, TILE, TILE, 4)
PLAYER_BY_DIR = {4: 37, 3: 38, 1: 39, 2: 40}  # down, up, left, right
ITEM_TEX = {1: 43, 2: 44, 3: 45, 4: 46}
# mob class -> base texture (zombie/skeleton/cow); deeper floors get a tint
MOB_TEX = {0: 47, 2: 48, 1: 49, 3: 50, 4: 50}
FLOOR_TINT = {0: None, 1: (170, 140, 220), 2: (150, 200, 150)}

VIEW_W, VIEW_H = 29, 19   # tiles
SCALE = 3
HUD_H = 46 * SCALE // 3 * 3


def blit(canvas, t, r, c, tint=None):
    tile = tex[t].astype(np.float32)
    if tint is not None:
        tile[..., :3] *= np.array(tint, np.float32) / 255.0
    a = tile[..., 3:4] / 255.0
    y, x = r * TILE, c * TILE
    dst = canvas[y:y + TILE, x:x + TILE]
    dst[:] = dst * (1 - a) + tile[..., :3] * a


def render(fr, idx):
    pr, pc = int(fr["posr"]), int(fr["posc"])
    r0 = min(max(pr - VIEW_H // 2, 0), MAP - VIEW_H)
    c0 = min(max(pc - VIEW_W // 2, 0), MAP - VIEW_W)
    canvas = np.zeros((VIEW_H * TILE, VIEW_W * TILE, 3), np.float32)
    for r in range(VIEW_H):
        for c in range(VIEW_W):
            wr, wc = r0 + r, c0 + c
            blit(canvas, int(fr["map"][wr, wc]), r, c)
            it = int(fr["item"][wr, wc])
            if it in ITEM_TEX:
                blit(canvas, ITEM_TEX[it], r, c)
    for j, m in enumerate(fr["mob"]):
        mask, mr, mc = int(m[0]), int(m[2]), int(m[3])
        if mask and r0 <= mr < r0 + VIEW_H and c0 <= mc < c0 + VIEW_W:
            blit(canvas, MOB_TEX.get(j // 3, 47), mr - r0, mc - c0,
                 tint=FLOOR_TINT.get(int(fr["level"])))
    blit(canvas, PLAYER_BY_DIR.get(int(fr["dir"]), 37), pr - r0, pc - c0)
    # darkness: per-cell light map (caves are dark except near torches)
    lq = fr["light_q"][r0:r0 + VIEW_H, c0:c0 + VIEW_W].astype(np.float32)
    lq = np.clip(lq / 255.0, 0.12, 1.0)
    canvas *= np.kron(lq, np.ones((TILE, TILE)))[..., None]

    img = Image.fromarray(canvas.clip(0, 255).astype(np.uint8))
    img = img.resize((VIEW_W * TILE * SCALE, VIEW_H * TILE * SCALE),
                     Image.NEAREST)
    hud = Image.new("RGB", (img.width, HUD_H), (24, 24, 28))
    d = ImageDraw.Draw(hud)
    inv = fr["inv"]
    floor_names = ["surface", "dungeon", "gnomish mines", "sewers", "vault",
                   "troll mines", "fire realm", "ice realm", "graveyard"]
    line1 = (f"floor: {floor_names[int(fr['level'])]}    "
             f"hp {fr['health']:.0f}  food {int(fr['food'])}  "
             f"water {int(fr['drink'])}  energy {int(fr['energy'])}")
    line2 = (f"wood {int(inv[0])}  stone {int(inv[1])}  coal {int(inv[2])}  "
             f"iron {int(inv[3])}  diamond {int(inv[4])}  "
             f"pickaxe T{int(inv[6])}  sword T{int(inv[7])}  "
             f"achievements {int(fr['ach_count'])}")
    d.text((10, 8), line1, fill=(235, 235, 225))
    d.text((10, 26), line2, fill=(180, 185, 175))
    out = Image.new("RGB", (img.width, img.height + HUD_H))
    out.paste(img, (0, 0))
    out.paste(hud, (0, img.height))
    return out


def main():
    frames_path, outdir = sys.argv[1], Path(sys.argv[2])
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    end = int(sys.argv[4]) if len(sys.argv) > 4 else None
    outdir.mkdir(parents=True, exist_ok=True)
    frames = np.fromfile(frames_path, dtype=frame_dtype)
    print(f"{len(frames)} frames, itemsize {frame_dtype.itemsize}")
    sel = frames[start:end]
    for i, fr in enumerate(sel):
        render(fr, i).save(outdir / f"f{i:05d}.png")
        if i % 200 == 0:
            print(f"{i}/{len(sel)}")
    print("done", len(sel))


if __name__ == "__main__":
    main()
