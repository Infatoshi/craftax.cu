"""Trim dead standalone-harness code from the vendored craftax.cu.

Mechanical transform: keeps lines 1..GAME_END (device game logic) verbatim,
splits the harness tail into top-level chunks (preprocessor-depth aware),
and keeps only chunks reachable from the symbols binding.cu references.
Everything else (standalone PPO trainer, megakernel, bench/play/gradcheck
modes, main) is deleted.
"""
import re
import sys

SRC = sys.argv[1]
DST = sys.argv[2]
GAME_END = 8628  # last line of the game-logic region (1-indexed)

# Symbols the PufferLib wrapper (binding.cu) references, plus init/global
# entry points it reaches through them.
ROOTS = {
    "cu_launch_reset_list_warp", "cu_launch_step_run", "cu_step_launch",
    "cu_vec_free", "cu_vec_init", "cu_copy_back", "cu_print_logs",
    "k_encode", "k_encode_tail", "k_env_init", "k_reset_list",
    "k_spawn_tail", "k_step", "k_step_run", "k_global_init",
    "g_reset_count", "g_craftax_reset_pool", "g_craftax_reset_pool_ready",
    "g_craftax_reset_pool_size",
}

lines = open(SRC).read().splitlines(keepends=True)
head = lines[:GAME_END]
tail = lines[GAME_END:]

# --- split tail into top-level chunks ---
chunks = []  # (start_idx, end_idx_exclusive, text)
starts = []
depth = 0
brace = 0
paren = 0

def code_only(line):
    """Strip string/char literals and // comments so delimiter counting
    sees only code."""
    line = re.sub(r'\\.', '', line)          # escaped chars first
    line = re.sub(r'"[^"]*"', '""', line)
    line = re.sub(r"'[^']*'", "''", line)
    line = re.sub(r'//.*', '', line)
    line = re.sub(r'/\*.*?\*/', '', line)
    return line

for i, l in enumerate(tail):
    s = l.strip()
    code = code_only(l)
    if s.startswith("#if"):
        if depth == 0 and brace == 0 and paren == 0:
            starts.append(i)
        depth += 1
        continue
    if s.startswith("#endif"):
        depth -= 1
        continue
    if depth > 0:
        continue
    if brace == 0 and paren == 0 and l[:1] not in (" ", "\t", "\n", "") \
            and not s.startswith("//"):
        # new top-level construct begins here unless it's a continuation
        if not s.startswith("}") and not s.startswith(")"):
            starts.append(i)
    if brace == 0:
        paren += code.count("(") - code.count(")")
    brace += code.count("{") - code.count("}")
starts = sorted(set(starts))
for j, st in enumerate(starts):
    en = starts[j + 1] if j + 1 < len(starts) else len(tail)
    chunks.append([st, en, "".join(tail[st:en])])
# a bare `template <...>` header belongs to the following definition
merged = []
pending = None
for c in chunks:
    body = c[2].strip()
    if body.startswith("template") and "{" not in body and ";" not in body:
        pending = c if pending is None else [pending[0], c[1], pending[2] + c[2]]
        continue
    if pending is not None:
        c = [pending[0], c[1], pending[2] + c[2]]
        pending = None
    merged.append(c)
if pending is not None:
    merged.append(pending)
chunks = merged

# --- extract declared names per chunk ---
SYM = re.compile(r"\b((?:k|cu|g|cf)_\w+|Cu[A-Z]\w+|Craftax\w*|CRAFTAX_\w+|CF_\w+|CU_\w+)\b")

def declared(text):
    """Names this chunk truly defines: the signature's function name, #define
    names, global variable names, and typedef/struct names. Function calls
    inside the body must NOT count."""
    names = set()
    # signature region: up to the first '{' or first ';'
    cut = len(text)
    for ch in ("{", ";"):
        p = text.find(ch)
        if p != -1:
            cut = min(cut, p + 1)
    sig = text[:cut]
    for m in re.finditer(r"#define\s+(\w+)", text):
        names.add(m.group(1))
    QUAL = {"__launch_bounds__", "CRAFTAX_STEP_LB", "CU_CHECK", "if", "while",
            "for", "switch", "sizeof", "defined"}
    for m in re.finditer(r"\b(\w+)\s*\(", sig):  # first non-qualifier = def name
        if m.group(1) not in QUAL:
            names.add(m.group(1))
            break
    if "(" not in sig:  # global variable declaration
        for m in re.finditer(r"\b(\w+)\s*(?:\[[^\]]*\])?\s*(?:=|;)", sig):
            names.add(m.group(1))
    for m in re.finditer(r"\b(?:struct|typedef struct|class)\s+(\w+)", sig):
        names.add(m.group(1))
    m = re.search(r"}\s*(\w+)\s*;", text)  # typedef struct {...} Name;
    if m:
        names.add(m.group(1))
    return names

decls = [declared(c[2]) for c in chunks]

# comment-only / preprocessor-only / blank chunks: keep unconditionally
def trivial(text):
    for l in text.splitlines():
        s = l.strip()
        if s and not s.startswith("//") and not s.startswith("#include"):
            return False
    return True

keep = [trivial(c[2]) for c in chunks]
refs = set(ROOTS)
changed = True
while changed:
    changed = False
    for k, (c, d) in enumerate(zip(chunks, decls)):
        if keep[k]:
            continue
        if d & refs:
            keep[k] = True
            # add prefixed identifiers this chunk references
            refs |= set(SYM.findall(c[2]))
            changed = True

text = "".join(head + [c[2] for k, c in zip(keep, chunks) if k])

# --- post-pass: cosmetics the chunk dropper can't see ---
# includes orphaned by the trainer removal
if text.count("cublas") <= 1:
    text = text.replace('#include <cublas_v2.h>\n', "")
if "nv_bfloat16" not in text.replace("#include <cuda_bf16.h>", ""):
    text = text.replace('#include <cuda_bf16.h>\n', "")
# the section banner describing the deleted standalone policy/trainer
i0 = text.find("// Batched policy + trainer kernels")
if i0 != -1:
    b0 = text.rfind("// ============", 0, i0)
    b1 = text.find("// ============", i0)
    b1 = text.find("\n", b1) + 1
    text = text[:b0] + (
        "// Run-mode step kernels: batched gameplay step + reset-flag\n"
        "// record, actions provided by the caller (the vec / the policy).\n"
    ) + text[b1:]
open(DST, "w").write(text)
out = text.splitlines(keepends=True)
kept = sum(len(c[2].splitlines()) for k, c in zip(keep, chunks) if k)
drop = sum(len(c[2].splitlines()) for k, c in zip(keep, chunks) if not k)
print(f"chunks={len(chunks)} kept_lines={GAME_END + kept} dropped_lines={drop}")
dropped_names = [sorted(d)[:1] for k, d in zip(keep, decls) if not k and d]
print("sample dropped:", [n[0] for n in dropped_names[:20]])
print("--- kept harness chunks > 30 lines ---")
for k, (c, d) in zip(keep, zip(chunks, decls)):
    n = len(c[2].splitlines())
    if k and n > 30:
        print(f"{n:5d}  {sorted(d)[:2]}")
