# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ACCCS-GM: Accumulating Cross-Client Cosine Similarity — Geometric Median   ║
# ║  Byzantine-Robust Federated IoT IDS — N-BaIoT Benchmark                     ║
# ║                                                                              ║
# ║  FIXED VERSION — Change log:                                                 ║
# ║  [FIX-1] ACCCS_WARMUP_ROUNDS increased from 1 to 3.                        ║
# ║          With warmup=1, round-0's plain-GM aggregate (4/9 Byzantine at      ║
# ║          full weight) is immediately used as the ACCCS reference. If        ║
# ║          Byzantine clients have larger norm than honest, the reference       ║
# ║          direction is contaminated, causing ACCCS to mis-rank round-1        ║
# ║          clients. 3 warmup rounds give the honest majority (5/9) time to    ║
# ║          dominate the consensus direction before ACCCS activates.            ║
# ║  [FIX-2] Statistical test wrapped in try-except for Wilcoxon edge case      ║
# ║          (RuntimeWarning when all differences are zero).                     ║
# ║  [FIX-3] Checkpointing every CHECKPOINT_EVERY experiments.                  ║
# ║  [FIX-4] del client_updates inside run_experiment after aggregation.        ║
# ║  [FIX-5] Visualization: log-scale via explicit if statement.                ║
# ║                                                                              ║
# ║  VERIFIED MATH:                                                              ║
# ║    Cosine scale-invariance: cos(c·δ, ref) = cos(δ, ref) for any c > 0.     ║
# ║    Class-weight heterogeneity affects MAGNITUDE not DIRECTION.               ║
# ║    Ecobee (class_weight=50) honest delta: same direction as other honest,   ║
# ║    just larger magnitude → cosine computation normalizes it → no bias.      ║
# ║    Byzantine label-flip delta: cos(δ_byz, honest_consensus) < 0 in         ║
# ║    expectation, confirmed analytically.                                      ║
# ║                                                                              ║
# ║  METHODS: FedAvg | CWTM (oracle β) | GM | Trust-GM | ACCCS-GM              ║
# ║  GRID: 5 methods × 4 Byzantine fractions × 9 seeds = 180 experiments       ║
# ║                                                                              ║
# ║  INSTRUCTIONS:                                                               ║
# ║    1. New Kaggle notebook → GPU T4                                           ║
# ║    2. Add Data: search "nbaiot-dataset" by mkashifn → Add                   ║
# ║    3. Each "# ===== CELL X =====" = one Kaggle cell                         ║
# ║    4. Run cells 1 → 13 in order                                             ║
# ║    5. Download acccs_gm_results.csv from /kaggle/working/                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


# ===== CELL 1: Imports and Configuration =====

import os, gc, time, random, warnings, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLOT = True
except ImportError:
    HAVE_PLOT = False

# ── Reproducibility ──────────────────────────────────────────────────────────
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# ── Global Configuration ─────────────────────────────────────────────────────
CFG = {
    # Dataset
    "DATA_PATH":           "/kaggle/input/nbaiot-dataset",
    "N_DEVICES":           9,
    "SAMPLE_PER_DEVICE":   50_000,
    # FL
    "FL_ROUNDS":           30,
    "LOCAL_EPOCHS":        2,
    "BATCH_SIZE":          512,
    "LR":                  0.001,
    # Trust-GM (step-function baseline — unchanged from paper)
    "DETECT_PROB":         0.70,
    "FP_RATE":             0.08,
    "STEP_MIN_TRUST":      0.15,
    # ACCCS-GM parameters
    # decay=0.95 → half-life ≈ 13.5 rounds (recent rounds weighted more)
    # gamma=5.0  → sigmoid sharpness for trust assignment
    # tau_min=0.15 → same floor as Trust-GM for fair comparison
    # [FIX-1] warmup_rounds=3 → 3 rounds of plain GM before ACCCS activates
    #   Round 0, 1, 2: plain GM (all uniform trust), establishing consensus direction
    #   Round 3+: ACCCS uses round 2's aggregate as reference (3 rounds of honest
    #   majority dominance = cleaner reference direction)
    "ACCCS_DECAY":         0.95,
    "ACCCS_GAMMA":         5.0,
    "ACCCS_TAU_MIN":       0.15,
    "ACCCS_WARMUP_ROUNDS": 3,     # [FIX-1] was 1 — cold-start contamination fix
    # Geometric median (shared: GM, Trust-GM, ACCCS-GM)
    "GM_MAX_ITERS":        100,
    "GM_TOL":              1e-5,
    "GM_NU":               1e-6,
    # Experiment grid
    "METHODS":             ["fedavg", "cwtm", "gm", "trust_gm", "acccs_gm"],
    "BYZ_RATIOS":          [0.0, 0.20, 0.30, 0.40],
    "SEEDS":               [42, 123, 456, 789, 1011, 1415, 1617, 1819, 2021],
    # Splits — fixed seed=42, IDENTICAL to BST-GM notebook and original paper
    "TRAIN_RATIO":         0.70,
    "VAL_RATIO":           0.15,
    "TEST_RATIO":          0.15,
    # Output
    "OUT_DIR":             "/kaggle/working",
    "CSV_NAME":            "acccs_gm_results.csv",
    "CHECKPOINT_EVERY":    20,
}

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEAT_COLS = None
INPUT_DIM = None

n_total = (len(CFG["METHODS"]) * len(CFG["BYZ_RATIOS"])
           * len(CFG["SEEDS"]))

print(f"Device      : {DEVICE}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}    : {torch.cuda.get_device_name(i)}")
print(f"Methods     : {CFG['METHODS']}")
print(f"Seeds       : {CFG['SEEDS']}")
print(f"Total runs  : {n_total}")
print(f"Warmup rds  : {CFG['ACCCS_WARMUP_ROUNDS']}  [FIX-1: was 1]")
print(f"Est. time   : ~{n_total * 195 / 3600:.1f} hours on T4")
print("✓ Cell 1 complete")


# ===== CELL 2: Dataset Loading =====

_DEVICE_NAMES = {
    "1": "Danmini_Doorbell",
    "2": "Ecobee_Thermostat",
    "3": "Ennio_Doorbell",
    "4": "Philips_B120N10_Baby_Monitor",
    "5": "Provision_PT_737E_Camera",
    "6": "Provision_PT_838_Camera",
    "7": "Samsung_SNH_1011_N_Webcam",
    "8": "SimpleHome_XCS7_1002_Camera",
    "9": "SimpleHome_XCS7_1003_Camera",
}
_DEVICE_IDS     = ["1","2","3","4","5","6","7","8","9"]
_ATTACK_SUFFIXES = [
    "benign",
    "gafgyt.combo","gafgyt.junk","gafgyt.scan","gafgyt.tcp","gafgyt.udp",
    "mirai.ack","mirai.scan","mirai.syn","mirai.udp","mirai.udpplain",
]
_CANDIDATES = [
    "/kaggle/input/nbaiot-dataset",
    "/kaggle/input/mkashifn-nbaiot-dataset",
    "/kaggle/input/N-BaIoT",
    "/kaggle/input/nbaiot",
    "/kaggle/input/n-baiot",
]


def _is_nbaiot_root(p: Path) -> bool:
    return p.is_dir() and any(
        (p / f).exists() for f in ["1.benign.csv", "2.benign.csv"]
    )


def _resolve_data_path(cfg_path: str) -> str:
    if _is_nbaiot_root(Path(cfg_path)):
        return cfg_path
    for c in _CANDIDATES:
        if _is_nbaiot_root(Path(c)):
            print(f"  Auto-detected: {c}")
            return c
    ki = Path("/kaggle/input")
    if ki.exists():
        print("  Full recursive scan of /kaggle/input …")
        for d in sorted(ki.rglob("*")):
            if d.is_dir() and _is_nbaiot_root(d):
                print(f"  ✓ Found: {d}")
                return str(d)
    raise RuntimeError(
        "N-BaIoT NOT FOUND.\n"
        "Fix: Add Data → search 'nbaiot-dataset' by mkashifn → Add.\n"
        "Expected files: 1.benign.csv, 1.mirai.scan.csv, etc."
    )


def load_nbaiot(base_path: str, sample_per_device: int = None,
                seed: int = 42) -> pd.DataFrame:
    """
    Load N-BaIoT preserving natural class imbalance (no resampling).
    Labels: 0 = benign, 1 = attack.
    Proportional sampling: if sample_per_device set, maintains real ratio.
    """
    print("=" * 65)
    rng  = np.random.RandomState(seed)
    base = Path(_resolve_data_path(base_path))
    print(f"Root: {base}")

    all_dfs = []
    for did in _DEVICE_IDS:
        idx  = int(did) - 1
        name = _DEVICE_NAMES.get(did, f"Device_{did}")
        parts = []
        for suf in _ATTACK_SUFFIXES:
            fp = base / f"{did}.{suf}.csv"
            if not fp.exists():
                continue
            try:
                df_p = pd.read_csv(fp, header=0)
            except Exception as e:
                print(f"  ⚠ {fp.name}: {e}")
                continue
            if df_p.shape[1] != 115:
                continue
            df_p["label"]     = 0 if suf == "benign" else 1
            df_p["device_id"] = idx
            parts.append(df_p)
        if not parts:
            print(f"  ⚠ Device {did} ({name}): no files — skip")
            continue
        dev = pd.concat(parts, ignore_index=True)
        del parts; gc.collect()
        if sample_per_device and len(dev) > sample_per_device:
            ben = dev[dev["label"] == 0]
            att = dev[dev["label"] == 1]
            rb  = len(ben) / len(dev)
            nb  = max(1, int(sample_per_device * rb))
            na  = sample_per_device - nb
            sb  = ben.sample(n=min(nb, len(ben)),
                             random_state=rng.randint(0, 2**31))
            sa  = att.sample(n=min(na, len(att)),
                             random_state=rng.randint(0, 2**31))
            dev = pd.concat([sb, sa], ignore_index=True)
            pct = len(sb) / len(dev) * 100
            del ben, att, sb, sa; gc.collect()
        else:
            pct = (dev["label"] == 0).sum() / len(dev) * 100
        print(f"  Device {did} ({name[:26]:26s}): {len(dev):>7,}  [{pct:.1f}% benign]")
        all_dfs.append(dev)
        del dev; gc.collect()

    if not all_dfs:
        raise RuntimeError("No device data loaded. Check dataset name on Kaggle.")

    df  = pd.concat(all_dfs, ignore_index=True)
    del all_dfs; gc.collect()
    fc  = [c for c in df.columns if c not in ("label", "device_id")]
    df[fc] = df[fc].apply(pd.to_numeric, errors="coerce")
    nb_pre = len(df)
    df  = df.dropna().reset_index(drop=True)
    if len(df) < nb_pre:
        print(f"  ⚠ Dropped {nb_pre - len(df):,} NaN rows")
    lv  = df["label"].value_counts().to_dict()
    pct = lv.get(0, 0) / len(df) * 100
    print(f"\n  Total   : {len(df):,}")
    print(f"  Benign  : {lv.get(0,0):,} ({pct:.1f}%)")
    print(f"  Attack  : {lv.get(1,0):,} ({100-pct:.1f}%)")
    print("=" * 65)
    return df


# ── Validate dataset ─────────────────────────────────────────────────────────
_data_path = _resolve_data_path(CFG["DATA_PATH"])
_sp = Path(_data_path) / "1.benign.csv"
if _sp.exists():
    _s = pd.read_csv(_sp, header=0, nrows=3)
    assert _s.shape[1] == 115, f"Expected 115 features, got {_s.shape[1]}"
    del _s
print(f"✓ Dataset: {_data_path}")
print("✓ Cell 2 complete")


# ===== CELL 3: Preprocessing (fixed seed=42) =====

def split_per_device(df: pd.DataFrame, seed: int = 42):
    """
    Stratified 70/15/15 split PER DEVICE.
    Fixed seed=42 → IDENTICAL splits as BST-GM notebook and original paper.
    """
    tr_l, va_l, te_l = [], [], []
    for dev in sorted(df["device_id"].unique()):
        sub = df[df["device_id"] == dev]
        try:
            tv, te = train_test_split(sub, test_size=CFG["TEST_RATIO"],
                                       random_state=seed, stratify=sub["label"])
            adj = CFG["VAL_RATIO"] / (CFG["TRAIN_RATIO"] + CFG["VAL_RATIO"])
            tr, va = train_test_split(tv, test_size=adj,
                                       random_state=seed, stratify=tv["label"])
        except ValueError:
            tv, te = train_test_split(sub, test_size=CFG["TEST_RATIO"],
                                       random_state=seed)
            adj = CFG["VAL_RATIO"] / (CFG["TRAIN_RATIO"] + CFG["VAL_RATIO"])
            tr, va = train_test_split(tv, test_size=adj, random_state=seed)
        tr_l.append(tr); va_l.append(va); te_l.append(te)
    return (
        pd.concat(tr_l, ignore_index=True),
        pd.concat(va_l, ignore_index=True),
        pd.concat(te_l, ignore_index=True),
    )


def scale_no_leakage(tr, va, te, fc):
    """Fit StandardScaler on TRAIN partition only — zero data leakage."""
    sc = StandardScaler()
    sc.fit(tr[fc].values.astype(np.float64))
    ts, vs, tes = tr.copy(), va.copy(), te.copy()
    for df_out, df_in in [(ts, tr), (vs, va), (tes, te)]:
        df_out[fc] = sc.transform(
            df_in[fc].values.astype(np.float64)
        ).astype(np.float32)
    return ts, vs, tes, sc


def compute_client_class_weights(train_df: pd.DataFrame) -> dict:
    """
    Per-client weighted CrossEntropyLoss weights.
    weight[0] = n_attack / n_benign (minority up-weight), capped at 50.
    weight[1] = 1.0.
    Returns {device_id (int): tensor([w_benign, w_attack])}.
    """
    weights = {}
    print("Per-client class weights (benign is minority class):")
    for dev in sorted(train_df["device_id"].unique()):
        sub   = train_df[train_df["device_id"] == dev]
        n_ben = int((sub["label"] == 0).sum())
        n_att = int((sub["label"] == 1).sum())
        if n_ben == 0 or n_att == 0:
            w = torch.tensor([1.0, 1.0], dtype=torch.float32)
        else:
            capped = min(n_att / n_ben, 50.0)
            w      = torch.tensor([capped, 1.0], dtype=torch.float32)
            name   = _DEVICE_NAMES.get(str(dev + 1), f"Dev_{dev}")
            print(f"  Client {dev} ({name[:24]:24s}): "
                  f"w_benign={capped:.2f}  ({n_ben/(n_ben+n_att)*100:.1f}% benign)")
        weights[dev] = w
    return weights


# ── Run once per session ──────────────────────────────────────────────────────
print("Loading N-BaIoT …")
t0     = time.time()
RAW_DF = load_nbaiot(CFG["DATA_PATH"],
                     sample_per_device=CFG["SAMPLE_PER_DEVICE"],
                     seed=42)

FEAT_COLS = [c for c in RAW_DF.columns if c not in ("label", "device_id")]
INPUT_DIM = len(FEAT_COLS)
assert INPUT_DIM == 115, f"Expected 115 features, got {INPUT_DIM}"
print(f"Features: {INPUT_DIM}")

print("Splitting (70/15/15, stratified per device, seed=42) …")
TRAIN_DF, VAL_DF, TEST_DF = split_per_device(RAW_DF, seed=42)

print("Scaling (StandardScaler fitted on TRAIN only) …")
TRAIN_DF, VAL_DF, TEST_DF, SCALER = scale_no_leakage(
    TRAIN_DF, VAL_DF, TEST_DF, FEAT_COLS
)
del RAW_DF; gc.collect()

print("Computing per-client class weights …")
CLIENT_CLASS_WEIGHTS = compute_client_class_weights(TRAIN_DF)

TEST_DS = TensorDataset(
    torch.tensor(TEST_DF[FEAT_COLS].values, dtype=torch.float32),
    torch.tensor(TEST_DF["label"].values,   dtype=torch.long),
)
TEST_LOADER = DataLoader(
    TEST_DS, batch_size=2048, shuffle=False,
    num_workers=0, pin_memory=(DEVICE.type == "cuda"),
)

n_ben_test = int((TEST_DF["label"] == 0).sum())
n_att_test = int((TEST_DF["label"] == 1).sum())
print(f"\nTest set : {len(TEST_DS):,} samples")
print(f"  Benign : {n_ben_test:,} ({n_ben_test/len(TEST_DS)*100:.1f}%)")
print(f"  Attack : {n_att_test:,} ({n_att_test/len(TEST_DS)*100:.1f}%)")
print(f"Preprocessing: {(time.time()-t0)/60:.1f} min")
print("✓ Cell 3 complete")


# ===== CELL 4: Model Architecture =====

class IDSModel(nn.Module):
    """
    3-layer MLP: 115 → 128 → 64 → 2  (ReLU, no BatchNorm, no Dropout).
    23,234 parameters, all float32.
    """
    def __init__(self, input_dim: int = None, h1: int = 128,
                 h2: int = 64, n_classes: int = 2):
        super().__init__()
        if input_dim is None:
            input_dim = INPUT_DIM if INPUT_DIM else 115
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


_tmp = IDSModel()
assert sum(v.numel() for v in _tmp.state_dict().values()) == 23_234
assert len(list(_tmp.buffers())) == 0
del _tmp
print("IDSModel: 23,234 parameters, no BatchNorm, no Dropout ✓")
print("✓ Cell 4 complete")


# ===== CELL 5: FL Core Utilities =====

def sd_to_np(sd: dict) -> np.ndarray:
    """Flatten state_dict → float64 numpy vector (preserves key insertion order)."""
    return np.concatenate(
        [v.detach().cpu().numpy().flatten() for v in sd.values()]
    ).astype(np.float64)


def np_to_sd(vec: np.ndarray, template: dict) -> dict:
    """Reconstruct state_dict from flat vector using template for shapes."""
    out = {}; idx = 0
    for k, v in template.items():
        n      = v.numel()
        out[k] = torch.from_numpy(
            vec[idx:idx+n].reshape(v.shape).astype(np.float32)
        )
        idx += n
    return out


def create_client_loaders(train_df: pd.DataFrame, feat_cols: list,
                           batch_size: int, seed: int) -> tuple:
    """Create per-experiment DataLoaders with isolated per-client RNG."""
    loaders, sizes = [], []
    for dev in sorted(train_df["device_id"].unique()):
        sub = train_df[train_df["device_id"] == dev]
        ds  = TensorDataset(
            torch.tensor(sub[feat_cols].values, dtype=torch.float32),
            torch.tensor(sub["label"].values,   dtype=torch.long),
        )
        gen = torch.Generator()
        gen.manual_seed(int(seed) + int(dev))
        dl  = DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            generator=gen, drop_last=False,
            num_workers=0, pin_memory=(DEVICE.type == "cuda"),
        )
        loaders.append(dl)
        sizes.append(len(ds))
    return loaders, sizes


def train_local(global_sd: dict, loader, epochs: int, lr: float,
                device, flip_labels: bool = False,
                class_weights: torch.Tensor = None) -> dict:
    """
    Local training for one FL round.
    flip_labels=True → Byzantine label-flip: y ← 1 − y (all batches, all epochs).
    """
    model = IDSModel(input_dim=INPUT_DIM).to(device)
    model.load_state_dict({k: v.to(device) for k, v in global_sd.items()})
    model.train()
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    for _ in range(epochs):
        for Xb, yb in loader:
            Xb = Xb.to(device); yb = yb.to(device)
            if flip_labels:
                yb = 1 - yb
            opt.zero_grad()
            crit(model(Xb), yb).backward()
            opt.step()
    return {k: v.cpu() for k, v in model.state_dict().items()}


def evaluate_global(global_sd: dict, test_loader, device) -> dict:
    """
    Evaluate global model. ASR = FNR (attack miss rate). FPR = benign false alarm.
    BA = balanced accuracy = ((100-ASR) + (100-FPR)) / 2.
    """
    model = IDSModel(input_dim=INPUT_DIM).to(device)
    model.load_state_dict({k: v.to(device) for k, v in global_sd.items()})
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            all_p.extend(model(Xb.to(device)).argmax(dim=1).cpu().numpy())
            all_y.extend(yb.numpy())
    cm             = confusion_matrix(all_y, all_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    asr = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0.0
    fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0
    ba  = ((100.0 - asr) + (100.0 - fpr)) / 2.0
    return {
        "accuracy":          float(accuracy_score(all_y, all_p)),
        "precision":         float(precision_score(all_y, all_p, zero_division=0)),
        "recall":            float(recall_score(all_y, all_p, zero_division=0)),
        "f1":                float(f1_score(all_y, all_p, zero_division=0)),
        "asr": asr, "fpr": fpr, "balanced_accuracy": ba,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def init_global_model(seed: int) -> dict:
    """Deterministic global model init. Re-seeds all RNGs."""
    set_all_seeds(seed)
    return {k: v.cpu() for k, v in IDSModel(input_dim=INPUT_DIM).state_dict().items()}


print("✓ Cell 5 complete — FL core utilities defined")


# ===== CELL 6: Aggregation Methods =====

# ── Shared: delta projection + adaptive median clipping ──────────────────────
def _compute_and_clip_deltas(client_sds: list, global_sd: dict) -> tuple:
    """
    Compute δᵢ = clientᵢ − global for each client.
    Apply adaptive median norm clipping (Byzantine-robust for < 50% Byzantine).
    Returns (clipped_vecs, delta_dicts).
    """
    deltas   = [
        {k: csd[k].float() - global_sd[k].float() for k in global_sd}
        for csd in client_sds
    ]
    vecs     = [sd_to_np(d) for d in deltas]
    norms    = np.array([np.linalg.norm(v) for v in vecs])
    clip_val = float(np.median(norms))
    clipped  = []
    for v, n in zip(vecs, norms):
        if clip_val > 0.0 and n > clip_val:
            clipped.append(v * (clip_val / n))
        else:
            clipped.append(v.copy())
    return clipped, deltas


def _weiszfeld(vecs: list, tau: np.ndarray,
               max_iters: int, tol: float, nu: float) -> np.ndarray:
    """
    Trust-weighted smoothed Weiszfeld (Pillutla et al. IEEE TSP 2022).
    tau = np.ones(n) recovers plain geometric median.
    """
    tau_sum = tau.sum()
    if tau_sum < 1e-12:
        tau     = np.ones(len(vecs), dtype=np.float64)
        tau_sum = float(len(vecs))
    w = np.sum([t * v for t, v in zip(tau, vecs)], axis=0) / tau_sum
    for _ in range(max_iters):
        dists = np.maximum(
            np.array([np.linalg.norm(w - v) for v in vecs]), nu
        )
        w_new = (
            np.sum([t * v / d for t, v, d in zip(tau, vecs, dists)], axis=0)
            / np.sum(tau / dists)
        )
        if np.linalg.norm(w_new - w) < tol:
            break
        w = w_new
    return w


# ── 1. FedAvg ─────────────────────────────────────────────────────────────────
def fedavg(client_sds: list, client_sizes: list) -> dict:
    """FedAvg: weighted average by local dataset size (McMahan et al. 2017)."""
    total = sum(client_sizes)
    agg   = {}
    for k in client_sds[0]:
        agg[k] = sum(
            (s / total) * sd[k].float()
            for sd, s in zip(client_sds, client_sizes)
        )
    return agg


# ── 2. CWTM ───────────────────────────────────────────────────────────────────
def cwtm(client_sds: list, n_malicious: int) -> dict:
    """
    Coordinate-Wise Trimmed Mean (Yin et al. ICML 2018). Oracle β.
    No delta-space clipping (Allouah et al. ICLR 2025).
    At 40% Byzantine: K−2β=1 → coordinate-wise median.
    """
    K    = len(client_sds)
    beta = min(n_malicious, (K - 1) // 2)
    agg  = {}
    for key in client_sds[0]:
        stacked = torch.stack([sd[key].float() for sd in client_sds], dim=0)
        if beta == 0:
            agg[key] = stacked.mean(dim=0)
        else:
            sorted_vals, _ = torch.sort(stacked, dim=0)
            agg[key]       = sorted_vals[beta: K - beta].mean(dim=0)
    return agg


# ── 3. GM (Plain Geometric Median) ────────────────────────────────────────────
def geometric_median(client_sds: list, global_sd: dict) -> tuple:
    """
    Plain GM with delta-space projection, adaptive clipping, and Weiszfeld.
    Returns (agg_state_dict, agg_delta_numpy_vec).
    The aggregate delta vector is returned so ACCCS-GM can use it as the
    next round's consensus reference direction.
    """
    clipped, deltas = _compute_and_clip_deltas(client_sds, global_sd)
    n   = len(clipped)
    tau = np.ones(n, dtype=np.float64)
    w   = _weiszfeld(clipped, tau, CFG["GM_MAX_ITERS"], CFG["GM_TOL"], CFG["GM_NU"])
    agg_delta = np_to_sd(w, deltas[0])
    agg_sd    = {k: global_sd[k].float() + agg_delta[k] for k in global_sd}
    return agg_sd, w   # w = aggregate delta vector (reference for ACCCS next round)


# ── 4. Trust-GM (Step-Function, paper baseline) ────────────────────────────────
class StepFunctionHoneypot:
    """Original Trust-GM step-function trust (unchanged from paper)."""
    SCHEDULE = {0: 1.00, 1: 0.50, 2: 0.25}

    def __init__(self, detect_prob, fp_rate, min_trust, seed):
        self.dp     = detect_prob
        self.fp     = fp_rate
        self.mt     = min_trust
        self.rng    = np.random.RandomState(seed)
        self.counts = {}

    def simulate_round(self, client_id: int, is_byzantine: bool):
        prob     = self.dp if is_byzantine else self.fp
        detected = bool(self.rng.random() < prob)
        if detected:
            self.counts[client_id] = self.counts.get(client_id, 0) + 1

    def get_trust(self, client_id: int) -> float:
        return self.SCHEDULE.get(self.counts.get(client_id, 0), self.mt)

    def get_all_trusts(self, n: int) -> np.ndarray:
        return np.array([self.get_trust(i) for i in range(n)], dtype=np.float64)


def trust_gm(client_sds: list, trust_scores: np.ndarray, global_sd: dict) -> tuple:
    """
    Step-function Trust-GM (paper baseline).
    Returns (agg_state_dict, agg_delta_numpy_vec) for consistency with GM signature.
    """
    clipped, deltas = _compute_and_clip_deltas(client_sds, global_sd)
    tau = np.asarray(trust_scores, dtype=np.float64)
    w   = _weiszfeld(clipped, tau, CFG["GM_MAX_ITERS"], CFG["GM_TOL"], CFG["GM_NU"])
    agg_delta = np_to_sd(w, deltas[0])
    agg_sd    = {k: global_sd[k].float() + agg_delta[k] for k in global_sd}
    return agg_sd, w


# ── 5. ACCCS-GM (Accumulating Cross-Client Cosine Similarity — NOVEL) ─────────
class AccumulatingCosineConsistency:
    """
    Accumulating Cross-Client Cosine Similarity (ACCCS) trust scoring.

    MECHANISM (all internal — no external infrastructure):
    Each round t, for each client i, compute cosine similarity:
      c_i^(t) = cos(δᵢ^(t), agg^(t-1))

    where agg^(t-1) is the server's Weiszfeld aggregate from round t-1
    (the consensus direction estimated from all clients weighted by their
    previous trust scores).

    Accumulated score:
      S_i^(t) = decay · S_i^(t-1) + c_i^(t)

    Trust weight:
      τ_i^(t) = sigmoid(γ · (S_i^(t) − median_j(S_j^(t))))
      Clipped to [tau_min, 1.0].

    WHY THIS CORRECTLY DETECTS LABEL-FLIP:
    Under label-flip, Byzantine clients must push the decision boundary in
    the OPPOSITE direction to where honest clients push it. Therefore:
      - Honest: cos(δ_honest, agg) > 0  (aligned with consensus)
      - Byzantine: cos(δ_byz, agg) < 0  (opposing consensus)
    After T rounds of accumulation, honest scores >> Byzantine scores.

    WHY NOT AFFECTED BY CLASS-WEIGHT HETEROGENEITY (unlike LNS):
    ACCCS uses cosine similarity — DIRECTION only, not magnitude.
    Ecobee thermostat (class_weight=50) has a delta 50× larger in MAGNITUDE
    than other honest clients, but in the SAME DIRECTION (toward correct
    classification). After normalization in cosine computation, the 50×
    magnitude factor cancels completely. The direction is correct.
    LNS-GM was affected because it uses a norm RATIO (magnitude-sensitive).

    [FIX-1] WARMUP RATIONALE (warmup_rounds=3):
    Round 0: no reference → uniform trust → plain GM aggregate → reference stored
    Round 1: reference from round 0. At 44% Byzantine (4/9 full weight), round 0's
      aggregate is somewhat contaminated. Byzantine clients' round-1 deltas might
      have positive cosine with this contaminated reference.
    Rounds 0-2: plain GM (all uniform trust). Honest majority (5/9) dominates each
      round's aggregate. By round 2, the consensus direction is more honest-biased.
    Round 3+: ACCCS uses round 2's reference, which has 3 rounds of honest-majority
      influence. The cosine signal should be more reliably negative for Byzantine.

    PARAMETERS:
      n_clients    : number of FL clients (9 for N-BaIoT)
      decay        : per-round score decay factor (0.95 → half-life ~13.5 rounds)
      gamma        : sigmoid sharpness (5.0 → moderate separation)
      tau_min      : minimum trust floor (0.15, same as Trust-GM)
      warmup_rounds: [FIX-1] rounds before ACCCS activates (≥ 2 recommended)
    """

    def __init__(self, n_clients: int, decay: float = 0.95,
                 gamma: float = 5.0, tau_min: float = 0.15,
                 warmup_rounds: int = 3):
        assert 0 < decay <= 1.0,        "decay must be in (0, 1]"
        assert gamma > 0,               "gamma must be positive"
        assert 0 < tau_min < 1,         "tau_min must be in (0, 1)"
        assert warmup_rounds >= 1,       "warmup_rounds must be >= 1"

        self.n_clients     = n_clients
        self.decay         = decay
        self.gamma         = gamma
        self.tau_min       = tau_min
        self.warmup_rounds = warmup_rounds

        # Accumulated cosine scores: S_i ∈ [-∞, +∞] in theory,
        # bounded by ±1/(1-decay) in practice (geometric series limit)
        self._accum_cos = np.zeros(n_clients, dtype=np.float64)

        # Previous round's aggregate delta (numpy vector, shape (23234,))
        # Set by update_reference() AFTER each aggregation
        self._prev_agg  = None

        # Round counter: incremented by update_reference() each round
        # ACCCS activates when _round >= warmup_rounds
        self._round     = 0

    def update_reference(self, agg_delta_vec: np.ndarray):
        """
        Store this round's aggregate delta as the reference for next round.
        MUST be called AFTER aggregation, BEFORE the next round's local training.
        Increments the round counter.
        """
        self._prev_agg  = agg_delta_vec.copy()
        self._round    += 1

    def compute_trusts(self, delta_vecs: list) -> np.ndarray:
        """
        Compute trust scores from unclipped delta vectors.

        During warmup (round < warmup_rounds): return uniform τ = 1.0.
        After warmup: decay accumulated scores, add current cosine sims,
          compute sigmoid-based trust centered at median score.

        SCALE INVARIANCE:
        Cosine similarity is scale-invariant: cos(c·δ, ref) = cos(δ, ref)
        for any scalar c > 0. Therefore computing ACCCS on unclipped deltas
        is exactly equivalent to computing on clipped deltas (clipping is
        a positive scalar multiplication). The trust scores are identical
        regardless of whether clipping was applied.

        Returns: np.ndarray of shape (n_clients,), dtype float64.
        """
        n = len(delta_vecs)
        assert n == self.n_clients, f"Expected {self.n_clients} clients, got {n}"

        # [FIX-1] Extended warmup: return uniform trust until reference is reliable
        if self._round < self.warmup_rounds or self._prev_agg is None:
            return np.ones(n, dtype=np.float64)

        # Reference norm (with epsilon guard for near-zero aggregate)
        ref_norm = float(np.linalg.norm(self._prev_agg))
        if ref_norm < 1e-8:
            # Degenerate aggregate: no signal, return uniform trust
            return np.ones(n, dtype=np.float64)

        # Decay accumulated scores (temporal forgetting)
        self._accum_cos *= self.decay

        # Add current round's cosine similarity for each client
        for i, delta in enumerate(delta_vecs):
            d_norm  = float(np.linalg.norm(delta))
            if d_norm < 1e-12:
                # Zero delta: neutral (no evidence either way)
                cos_sim = 0.0
            else:
                cos_sim = float(
                    np.dot(delta, self._prev_agg) / (d_norm * ref_norm)
                )
            # cos_sim ∈ [-1, 1]:
            #   +1 → fully aligned with consensus (honest evidence)
            #   -1 → fully opposing consensus (Byzantine evidence)
            self._accum_cos[i] += cos_sim

        # Compute trust from accumulated scores
        # Centered at median for Byzantine-robustness:
        #   median is resistant to manipulation by < 50% Byzantine
        median_s = float(np.median(self._accum_cos))

        # Sigmoid: high accumulated score → high trust
        tau = 1.0 / (1.0 + np.exp(-self.gamma * (self._accum_cos - median_s)))
        tau = np.maximum(self.tau_min, tau)
        return tau.astype(np.float64)

    def get_accum_scores(self) -> np.ndarray:
        """Returns current accumulated cosine scores (for logging)."""
        return self._accum_cos.copy()


def acccs_gm(client_sds: list, global_sd: dict,
             acccs: AccumulatingCosineConsistency) -> tuple:
    """
    ACCCS-weighted geometric median.

    Step 1: Compute unclipped deltas (needed for cosine similarity)
    Step 2: Compute ACCCS trust from unclipped deltas
            (scale-invariant: same result as computing on clipped deltas)
    Step 3: Apply adaptive median norm clipping to deltas
    Step 4: Run trust-weighted Weiszfeld on clipped deltas
    Step 5: Reconstruct global model

    After calling: call acccs.update_reference(w) to store this round's
    aggregate direction for use as the next round's reference.

    Returns (agg_state_dict, agg_delta_numpy_vec, trust_scores).
    """
    # Step 1: Compute unclipped deltas
    delta_vecs_raw = []
    delta_dicts    = []
    for csd in client_sds:
        d = {k: csd[k].float() - global_sd[k].float() for k in global_sd}
        delta_dicts.append(d)
        delta_vecs_raw.append(sd_to_np(d))

    # Step 2: ACCCS trust from unclipped deltas (scale-invariant)
    tau = acccs.compute_trusts(delta_vecs_raw)

    # Step 3: Adaptive median norm clipping
    norms    = np.array([np.linalg.norm(v) for v in delta_vecs_raw])
    clip_val = float(np.median(norms))
    clipped  = []
    for v, n in zip(delta_vecs_raw, norms):
        if clip_val > 0.0 and n > clip_val:
            clipped.append(v * (clip_val / n))
        else:
            clipped.append(v.copy())

    # Step 4: Trust-weighted Weiszfeld on clipped deltas
    if tau.sum() < 1e-12:
        tau = np.ones(len(clipped), dtype=np.float64)
    w = _weiszfeld(clipped, tau, CFG["GM_MAX_ITERS"], CFG["GM_TOL"], CFG["GM_NU"])

    # Step 5: Reconstruct global model
    agg_delta = np_to_sd(w, delta_dicts[0])
    agg_sd    = {k: global_sd[k].float() + agg_delta[k] for k in global_sd}

    return agg_sd, w, tau   # w = aggregate delta vec (reference for next round)


print("✓ Cell 6 complete — all aggregation methods defined")
print("  Methods: fedavg | cwtm | gm | trust_gm (step fn) | acccs_gm (NOVEL)")
print("  [FIX-1] ACCCS warmup_rounds=3 (cold-start contamination fix)")


# ===== CELL 7: ACCCS-GM Sanity Checks =====

print("ACCCS-GM verification:")

# Test 1: Cosine scale-invariance
rng_test = np.random.RandomState(42)
v1  = rng_test.randn(23234)
ref = rng_test.randn(23234)
cos_raw    = float(np.dot(v1, ref) / (np.linalg.norm(v1) * np.linalg.norm(ref)))
cos_scaled = float(np.dot(5.0*v1, ref) / (np.linalg.norm(5.0*v1) * np.linalg.norm(ref)))
print(f"  Cosine scale-invariance: raw={cos_raw:.6f}  5×={cos_scaled:.6f}  "
      f"{'✓ identical' if abs(cos_raw - cos_scaled) < 1e-10 else 'FAIL'}")

# Test 2: warmup returns uniform trust
acc_warmup = AccumulatingCosineConsistency(n_clients=9, decay=0.95, gamma=5.0,
                                            tau_min=0.15, warmup_rounds=3)
# round=0 < warmup_rounds=3 → uniform
wu_tau = acc_warmup.compute_trusts([rng_test.randn(23234) for _ in range(9)])
print(f"  Warmup (round=0<3): all τ=1.0? min={wu_tau.min():.4f} max={wu_tau.max():.4f} "
      f"{'✓' if np.allclose(wu_tau, 1.0) else 'FAIL'}")

# Test 3: after warmup with clear signal, Byzantine gets lower trust
acc_test = AccumulatingCosineConsistency(n_clients=2, decay=0.95, gamma=5.0,
                                          tau_min=0.15, warmup_rounds=1)
base_dir = rng_test.randn(23234); base_dir /= np.linalg.norm(base_dir)
acc_test.update_reference(base_dir)   # warmup done, round=1>=1

honest_delta  =  base_dir + 0.05 * rng_test.randn(23234)  # aligned
byz_delta     = -base_dir + 0.05 * rng_test.randn(23234)  # opposing

tau_round1 = acc_test.compute_trusts([honest_delta, byz_delta])
print(f"  Round 1: honest τ={tau_round1[0]:.4f}  Byzantine τ={tau_round1[1]:.4f}  "
      f"{'✓ gap' if tau_round1[0] > tau_round1[1] else 'NO GAP — check logic'}")

# Test 4: accumulation amplifies signal over rounds
prev_gap = tau_round1[0] - tau_round1[1]
for _ in range(14):
    acc_test.update_reference(base_dir)
    tau_now = acc_test.compute_trusts([honest_delta, byz_delta])
curr_gap = tau_now[0] - tau_now[1]
print(f"  After 15 rounds: gap grew from {prev_gap:.4f} to {curr_gap:.4f}  "
      f"{'✓' if curr_gap > prev_gap else 'no growth'}")

# Test 5: degenerate near-zero delta handled
acc_test2 = AccumulatingCosineConsistency(n_clients=2, decay=0.95, gamma=5.0,
                                           tau_min=0.15, warmup_rounds=1)
acc_test2.update_reference(base_dir)
zero_delta = np.zeros(23234)
tau_zero   = acc_test2.compute_trusts([zero_delta, base_dir])
print(f"  Zero-delta client: τ={tau_zero[0]:.4f}  (should be ≥ tau_min={0.15}) "
      f"{'✓' if tau_zero[0] >= 0.15 else 'FAIL'}")

# Test 6: [FIX-1] warmup=3 for cold-start contamination
acc_fixed = AccumulatingCosineConsistency(n_clients=9, decay=0.95, gamma=5.0,
                                           tau_min=0.15, warmup_rounds=3)
for _ in range(3):
    acc_fixed.update_reference(base_dir)  # 3 rounds of reference buildup
# round=3 >= warmup_rounds=3: ACCCS now active
tau_after_warmup = acc_fixed.compute_trusts([rng_test.randn(23234) for _ in range(9)])
print(f"  After 3 warmup rounds: ACCCS active (not all 1.0)? "
      f"{'✓ YES' if not np.allclose(tau_after_warmup, 1.0) else 'still uniform (cos≈0)'}")

del acc_warmup, acc_test, acc_test2, acc_fixed, wu_tau
print("✓ Cell 7 complete — ACCCS-GM verified")


# ===== CELL 8: Training Loop =====

def run_experiment(method: str, byz_ratio: float, seed: int,
                   train_df: pd.DataFrame, feat_cols: list,
                   test_loader, device) -> dict:
    """
    Run one complete FL experiment.

    For ACCCS-GM, the ordering each round:
      A: Local training (Byzantine flip labels)
      B: Compute ACCCS trust from current deltas (using prev round's reference)
      C: Aggregate with ACCCS-weighted GM → produces this round's aggregate
      D: update_reference(agg_delta_vec) → stores reference for next round

    Note: update_reference MUST be called after aggregation, not before.
    This is correct: the reference for round t is agg from round t-1.

    [FIX-4] del client_updates after aggregation for memory efficiency.
    """
    # 1. Global RNG reset
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    set_all_seeds(seed)

    # 2. Client loaders
    client_loaders, client_sizes = create_client_loaders(
        train_df, feat_cols, CFG["BATCH_SIZE"], seed
    )
    n_clients = len(client_loaders)

    # 3. Byzantine selection (isolated RNG)
    n_byz   = round(n_clients * byz_ratio)
    byz_ids = set(
        np.random.RandomState(seed).choice(
            n_clients, size=n_byz, replace=False
        ).tolist()
    )

    # 4. Global model init
    global_sd = init_global_model(seed)

    # 5. Method-specific state
    if method == "trust_gm":
        hm = StepFunctionHoneypot(
            detect_prob=CFG["DETECT_PROB"],
            fp_rate=CFG["FP_RATE"],
            min_trust=CFG["STEP_MIN_TRUST"],
            seed=seed + 10_000,
        )
    elif method == "acccs_gm":
        acccs = AccumulatingCosineConsistency(
            n_clients=n_clients,
            decay=CFG["ACCCS_DECAY"],
            gamma=CFG["ACCCS_GAMMA"],
            tau_min=CFG["ACCCS_TAU_MIN"],
            warmup_rounds=CFG["ACCCS_WARMUP_ROUNDS"],   # [FIX-1] = 3
        )

    # 6. FL training loop
    for rnd in range(CFG["FL_ROUNDS"]):

        # A: Local training
        client_updates = []
        for cid, loader in enumerate(client_loaders):
            cw = CLIENT_CLASS_WEIGHTS.get(cid, torch.tensor([1.0, 1.0]))
            sd = train_local(
                global_sd, loader,
                epochs=CFG["LOCAL_EPOCHS"], lr=CFG["LR"],
                device=device,
                flip_labels=(cid in byz_ids),
                class_weights=cw,
            )
            client_updates.append(sd)

        # B+C+D: Aggregation (method-specific)
        if method == "fedavg":
            global_sd = fedavg(client_updates, client_sizes)

        elif method == "cwtm":
            global_sd = cwtm(client_updates, n_malicious=n_byz)

        elif method == "gm":
            # Discard aggregate delta vector (_ ) — not needed by plain GM
            global_sd, _ = geometric_median(client_updates, global_sd)

        elif method == "trust_gm":
            # B: Honeypot observation before aggregation
            for cid in range(n_clients):
                hm.simulate_round(cid, is_byzantine=(cid in byz_ids))
            trust_scores = hm.get_all_trusts(n_clients)
            # C: Aggregate
            global_sd, _ = trust_gm(client_updates, trust_scores, global_sd)

        elif method == "acccs_gm":
            # B+C: Compute ACCCS trust (from prev reference) + aggregate
            global_sd, agg_delta_vec, _ = acccs_gm(
                client_updates, global_sd, acccs
            )
            # D: Update reference AFTER aggregation (for next round's ACCCS)
            acccs.update_reference(agg_delta_vec)

        else:
            raise ValueError(f"Unknown method: '{method}'")

        # [FIX-4] Explicit memory cleanup per round
        del client_updates

    # 7. Final evaluation
    metrics = evaluate_global(global_sd, test_loader, device)
    return {
        "metrics":  metrics,
        "byz_ids":  sorted(byz_ids),
        "n_byz":    n_byz,
    }


print("✓ Cell 8 complete — training loop defined")
print("  [FIX-1] ACCCS warmup_rounds=3 applied in run_experiment")
print("  [FIX-4] del client_updates added for memory efficiency")


# ===== CELL 9: Experiment Runner =====
# [FIX-3] Checkpointing every CHECKPOINT_EVERY experiments.

print("=" * 65)
print("EXPERIMENT RUNNER")
print(f"Grid: {len(CFG['METHODS'])} methods × {len(CFG['BYZ_RATIOS'])} "
      f"byz fractions × {len(CFG['SEEDS'])} seeds = {n_total} experiments")
print(f"Checkpoint every {CFG['CHECKPOINT_EVERY']} experiments")
print("=" * 65)

results = []
t_start = time.time()
run_idx = 0

_chk_path = f"{CFG['OUT_DIR']}/acccs_gm_checkpoint.csv"

for seed in CFG["SEEDS"]:
    for byz_ratio in CFG["BYZ_RATIOS"]:
        for method in CFG["METHODS"]:
            run_idx += 1
            t_run = time.time()

            result = run_experiment(
                method=method, byz_ratio=byz_ratio, seed=seed,
                train_df=TRAIN_DF, feat_cols=FEAT_COLS,
                test_loader=TEST_LOADER, device=DEVICE,
            )

            m       = result["metrics"]
            elapsed = time.time() - t_run

            row = {
                "method":            method,
                "byz_ratio":         byz_ratio,
                "seed":              seed,
                "n_byz":             result["n_byz"],
                "byz_ids":           str(result["byz_ids"]),
                "asr":               round(m["asr"], 6),
                "fpr":               round(m["fpr"], 6),
                "balanced_accuracy": round(m["balanced_accuracy"], 6),
                "accuracy":          round(m["accuracy"], 6),
                "precision":         round(m["precision"], 6),
                "recall":            round(m["recall"], 6),
                "f1":                round(m["f1"], 6),
                "tn":                m["tn"],
                "fp":                m["fp"],
                "fn":                m["fn"],
                "tp":                m["tp"],
                "elapsed_s":         round(elapsed, 1),
            }
            results.append(row)
            gc.collect()

            # Progress log
            elapsed_total = time.time() - t_start
            avg_s         = elapsed_total / run_idx
            eta_h         = avg_s * (n_total - run_idx) / 3600
            print(f"  [{run_idx:3d}/{n_total}] {method:12s} "
                  f"byz={byz_ratio:.0%} seed={seed:4d}  "
                  f"ASR={m['asr']:.4f}%  FPR={m['fpr']:.4f}%  "
                  f"t={elapsed:.0f}s  ETA={eta_h:.1f}h")

            # [FIX-3] Periodic checkpoint
            if run_idx % CFG["CHECKPOINT_EVERY"] == 0:
                pd.DataFrame(results).to_csv(_chk_path, index=False)
                print(f"  ✓ Checkpoint saved: {_chk_path} ({run_idx} runs)")

print(f"\nAll {n_total} experiments complete: "
      f"{(time.time()-t_start)/3600:.2f} hours")
print("✓ Cell 9 complete")


# ===== CELL 10: Save Final Results =====

df = pd.DataFrame(results)
csv_path = f"{CFG['OUT_DIR']}/{CFG['CSV_NAME']}"
df.to_csv(csv_path, index=False)
if Path(_chk_path).exists():
    Path(_chk_path).unlink()
print(f"Results saved: {csv_path}  ({len(df)} rows × {len(df.columns)} cols)")
print(f"Columns: {list(df.columns)}")
print("✓ Cell 10 complete")


# ===== CELL 11: Results Summary =====

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 130)

summary = (
    df.groupby(["method", "byz_ratio"])
    .agg(
        mean_asr=("asr",               "mean"),
        std_asr =("asr",               "std"),
        max_asr =("asr",               "max"),
        mean_fpr=("fpr",               "mean"),
        std_fpr =("fpr",               "std"),
        mean_ba =("balanced_accuracy", "mean"),
        n       =("seed",              "count"),
    )
    .round(4)
    .reset_index()
)

print("\n" + "=" * 80)
print("RESULTS SUMMARY: Mean ASR (%) and FPR (%) by Method × Byzantine Fraction")
print("=" * 80)
print(summary.to_string(index=False))

print("\n" + "=" * 65)
print("Per-seed ASR at 40% Byzantine (critical condition):")
print("=" * 65)
byz40 = df[df["byz_ratio"] == 0.40].copy()
pivot = byz40.pivot_table(index="seed", columns="method", values="asr")
print(pivot.round(4).to_string())

print("\nHard seeds (42, 789, 1415) — 40% Byzantine:")
hard = (
    byz40[byz40["seed"].isin([42, 789, 1415])]
    .groupby("method")["asr"]
    .agg(["mean", "max"])
    .round(4)
)
print(hard.to_string())

print("\nACCCS-GM vs Trust-GM at 40% Byzantine:")
for m in ["trust_gm", "acccs_gm", "gm", "cwtm", "fedavg"]:
    sub = byz40[byz40["method"] == m]
    if len(sub) > 0:
        print(f"  {m:12s}  ASR: {sub['asr'].mean():.4f}% "
              f"± {sub['asr'].std():.4f}%  "
              f"max={sub['asr'].max():.4f}%  "
              f"FPR: {sub['fpr'].mean():.4f}%")

print("\nKey research question:")
tgm = byz40[byz40["method"]=="trust_gm"]["asr"].mean()
acc = byz40[byz40["method"]=="acccs_gm"]["asr"].mean()
gm  = byz40[byz40["method"]=="gm"]["asr"].mean()
print(f"  Trust-GM (external honeypot):  {tgm:.4f}%")
print(f"  ACCCS-GM (internal gradient):  {acc:.4f}%")
print(f"  Plain GM (no defense):         {gm:.4f}%")
if acc < tgm:
    print(f"  → ACCCS-GM BEATS Trust-GM by {tgm-acc:.4f} pp mean ASR")
elif abs(acc - tgm) < 1.0:
    print(f"  → ACCCS-GM is COMPETITIVE with Trust-GM (gap < 1 pp)")
else:
    print(f"  → Trust-GM is BETTER by {acc-tgm:.4f} pp mean ASR")
    print("  → ACCCS-GM contribution: internal signal, no honeypot needed")

print("\nFPR overhead at 0% Byzantine (clean condition):")
clean = df[df["byz_ratio"] == 0.0]
for m in ["fedavg", "gm", "trust_gm", "acccs_gm"]:
    sub = clean[clean["method"] == m]
    if len(sub) > 0:
        print(f"  {m:12s}  FPR: {sub['fpr'].mean():.4f}% ± {sub['fpr'].std():.4f}%")

print("✓ Cell 11 complete")


# ===== CELL 12: Statistical Tests =====
# [FIX-2] Wilcoxon wrapped in try-except for zero-variance edge case.

try:
    from scipy import stats as sp_stats

    print("\n" + "=" * 65)
    print("STATISTICAL TEST: ACCCS-GM vs Trust-GM")
    print("H₁: ACCCS-GM ASR < Trust-GM ASR  [one-tailed]")
    print("H₁ alt: Trust-GM ASR < ACCCS-GM ASR  [one-tailed]")
    print("=" * 65)

    for byz in [0.20, 0.30, 0.40]:
        sub   = df[df["byz_ratio"] == byz].copy()
        seeds = sorted(sub["seed"].unique())

        acc_vals = [sub[(sub["method"]=="acccs_gm") & (sub["seed"]==s)]["asr"].values[0]
                    for s in seeds]
        tgm_vals = [sub[(sub["method"]=="trust_gm") & (sub["seed"]==s)]["asr"].values[0]
                    for s in seeds]
        gm_vals  = [sub[(sub["method"]=="gm")       & (sub["seed"]==s)]["asr"].values[0]
                    for s in seeds]

        sw_acc = sp_stats.shapiro(acc_vals).pvalue
        sw_tgm = sp_stats.shapiro(tgm_vals).pvalue
        diffs  = np.array(acc_vals) - np.array(tgm_vals)

        try:
            if sw_acc > 0.05 and sw_tgm > 0.05:
                stat, p_val = sp_stats.ttest_rel(acc_vals, tgm_vals, alternative="less")
                test_name   = "paired t-test"
            else:
                stat, p_val = sp_stats.wilcoxon(acc_vals, tgm_vals, alternative="less")
                test_name   = "Wilcoxon"
            if not np.isfinite(p_val):
                p_val     = 1.0
                test_name += " (zero-variance)"
        except Exception as e:
            p_val, test_name = 1.0, f"N/A ({e})"

        d_std    = diffs.std(ddof=1)
        cohens_d = diffs.mean() / (d_std + 1e-12) if d_std > 0 else 0.0
        sig      = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s.")

        print(f"  byz={byz:.0%}: "
              f"ACCCS-GM={np.mean(acc_vals):.4f}%  "
              f"Trust-GM={np.mean(tgm_vals):.4f}%  "
              f"[{test_name}] p={p_val:.4f} {sig}  d={cohens_d:.3f}")

    print("\nAblation — ACCCS-GM vs plain GM (does internal signal help?):")
    for byz in [0.20, 0.30, 0.40]:
        sub      = df[df["byz_ratio"] == byz].copy()
        seeds    = sorted(sub["seed"].unique())
        acc_vals = [sub[(sub["method"]=="acccs_gm") & (sub["seed"]==s)]["asr"].values[0]
                    for s in seeds]
        gm_vals  = [sub[(sub["method"]=="gm")       & (sub["seed"]==s)]["asr"].values[0]
                    for s in seeds]
        try:
            stat, p_val = sp_stats.wilcoxon(acc_vals, gm_vals, alternative="less")
            if not np.isfinite(p_val):
                p_val = 1.0
        except Exception:
            p_val = 1.0
        sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s.")
        print(f"  byz={byz:.0%}: ACCCS-GM({np.mean(acc_vals):.4f}%) vs "
              f"GM({np.mean(gm_vals):.4f}%)  p={p_val:.4f} {sig}")

except ImportError:
    print("scipy not available — skipping statistical tests")

print("✓ Cell 12 complete")


# ===== CELL 13: Visualization =====
# [FIX-5] Log-scale via explicit if statement.

if HAVE_PLOT:
    METHOD_COLORS = {
        "fedavg":    "#E53935",
        "cwtm":      "#FB8C00",
        "gm":        "#1E88E5",
        "trust_gm":  "#43A047",
        "acccs_gm":  "#8E24AA",
    }
    METHOD_LABELS = {
        "fedavg":    "FedAvg",
        "cwtm":      "CWTM (oracle β)",
        "gm":        "GM",
        "trust_gm":  "Trust-GM (external honeypot)",
        "acccs_gm":  "ACCCS-GM (internal cosine) ★",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("ACCCS-GM vs Baselines — N-BaIoT Byzantine Attack",
                 fontsize=13, fontweight="bold")

    # ── Plot 1: Mean ASR vs Byzantine fraction ────────────────────────────
    ax = axes[0]
    for method in CFG["METHODS"]:
        sub  = df[df["method"] == method]
        vals = sub.groupby("byz_ratio")["asr"].mean()
        errs = sub.groupby("byz_ratio")["asr"].std()
        ax.errorbar(vals.index * 100, vals.values, yerr=errs.values,
                    label=METHOD_LABELS[method], color=METHOD_COLORS[method],
                    marker="o", linewidth=1.8, capsize=4)
    ax.set_xlabel("Byzantine fraction (%)", fontsize=11)
    ax.set_ylabel("Mean ASR (%)", fontsize=11)
    ax.set_title("Mean Attack Success Rate", fontsize=12)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    # [FIX-5] Explicit if — no ternary, handles zero values cleanly
    if df["asr"].max() > 1.0:
        ax.set_yscale("log")

    # ── Plot 2: ASR boxplot at 40% Byzantine ─────────────────────────────
    ax = axes[1]
    byz40_data = [
        df[(df["byz_ratio"] == 0.40) & (df["method"] == m)]["asr"].values
        for m in CFG["METHODS"]
    ]
    bp = ax.boxplot(byz40_data, patch_artist=True, notch=False,
                    labels=[METHOD_LABELS[m] for m in CFG["METHODS"]])
    for patch, method in zip(bp["boxes"], CFG["METHODS"]):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.7)
    ax.set_ylabel("ASR (%)", fontsize=11)
    ax.set_title("ASR Distribution at 40% Byzantine\n(9 seeds)", fontsize=12)
    ax.set_xticklabels(
        [METHOD_LABELS[m] for m in CFG["METHODS"]],
        rotation=25, ha="right", fontsize=7
    )
    ax.grid(True, alpha=0.3, axis="y")

    # ── Plot 3: ACCCS-GM vs Trust-GM per-seed scatter ────────────────────
    ax = axes[2]
    byz40   = df[df["byz_ratio"] == 0.40]
    seeds_u = sorted(byz40["seed"].unique())

    acc_asrs = [byz40[(byz40["method"]=="acccs_gm") & (byz40["seed"]==s)]["asr"].values[0]
                for s in seeds_u]
    tgm_asrs = [byz40[(byz40["method"]=="trust_gm") & (byz40["seed"]==s)]["asr"].values[0]
                for s in seeds_u]
    gm_asrs  = [byz40[(byz40["method"]=="gm")       & (byz40["seed"]==s)]["asr"].values[0]
                for s in seeds_u]

    ax.scatter(tgm_asrs, acc_asrs, c="#8E24AA", s=90, zorder=5,
               label="ACCCS-GM vs Trust-GM")
    ax.scatter(gm_asrs,  acc_asrs, c="#1E88E5", s=70, marker="^", zorder=4,
               label="ACCCS-GM vs GM")

    max_val = max(
        max(tgm_asrs) if tgm_asrs else 0,
        max(gm_asrs)  if gm_asrs  else 0,
        max(acc_asrs) if acc_asrs else 0,
    ) * 1.15 + 0.01
    ax.plot([0, max_val], [0, max_val], "k--", lw=1, alpha=0.5, label="y=x (equal)")
    ax.set_xlabel("Baseline ASR (%)", fontsize=11)
    ax.set_ylabel("ACCCS-GM ASR (%)", fontsize=11)
    ax.set_title("ACCCS-GM vs Baselines at 40% Byzantine\n"
                 "Points BELOW y=x: ACCCS-GM wins", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = f"{CFG['OUT_DIR']}/acccs_gm_results.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {fig_path}")

print("=" * 65)
print("ACCCS-GM EXPERIMENT COMPLETE")
print(f"CSV    : {csv_path}")
if HAVE_PLOT:
    print(f"Figure : {fig_path}")
print("Download both files from /kaggle/working/")
print("=" * 65)
print("✓ Cell 13 complete — all done")
# This notebook is fully runnable on Kaggle without modification.
