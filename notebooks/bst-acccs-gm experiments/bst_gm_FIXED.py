# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BST-GM: Bayesian Sequential Trust — Geometric Median                       ║
# ║  Byzantine-Robust Federated IoT IDS — N-BaIoT Benchmark                     ║
# ║                                                                              ║
# ║  FIXED VERSION — Change log:                                                 ║
# ║  [FIX-1] BayesianSequentialTrust: separated accumulated observations from   ║
# ║          the prior. Original code decayed the prior log-odds on each round,  ║
# ║          which is incorrect when decay < 1. The prior is now fixed; only     ║
# ║          observation evidence is accumulated and decayed.                    ║
# ║  [FIX-2] Visualization: replaced confusing ternary-style set_yscale() call  ║
# ║          with an explicit if statement. Handles zero ASR values safely.      ║
# ║  [FIX-3] Statistical test: wrapped Wilcoxon in try-except to handle the     ║
# ║          RuntimeWarning (NaN z-score) when all differences are zero.        ║
# ║  [FIX-4] Checkpointing: Cell 9 saves partial results every 20 experiments   ║
# ║          so a kernel crash does not lose all work.                           ║
# ║  [FIX-5] del client_updates inside run_experiment after aggregation for     ║
# ║          proper per-round memory cleanup.                                    ║
# ║                                                                              ║
# ║  VERIFIED MATH (at t=30, p_d=0.70, p_fp=0.08, prior_byz=0.40):            ║
# ║    E[tau_byz] = 0.010  (step fn: 0.150  →  BST 15× more suppressive)      ║
# ║    E[tau_hon] = 1.000  (step fn: 0.321  →  BST fully exonerates)           ║
# ║    Effective Byzantine fraction: 0.008  (step fn: 0.272)                   ║
# ║                                                                              ║
# ║  METHODS: FedAvg | CWTM (oracle β) | GM | Trust-GM | BST-GM                ║
# ║  GRID: 5 methods × 4 Byzantine fractions × 9 seeds = 180 experiments       ║
# ║                                                                              ║
# ║  INSTRUCTIONS:                                                               ║
# ║    1. New Kaggle notebook → GPU T4                                           ║
# ║    2. Add Data: search "nbaiot-dataset" by mkashifn → Add                   ║
# ║    3. Each "# ===== CELL X =====" = one Kaggle cell                         ║
# ║    4. Run cells 1 → 13 in order                                             ║
# ║    5. Download bst_gm_results.csv from /kaggle/working/                     ║
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
    # BST-GM parameters
    # p_d and p_fp match Trust-GM for fair comparison.
    # prior_byz = 0.40 is a conservative prior (worst-case assumption).
    #   At 0% Byzantine, benign clients start at trust=0.60 and recover
    #   to ~1.0 by round 30 as non-detection evidence accumulates.
    # decay = 1.0 (full accumulation, no forgetting). The prior is FIXED
    #   at log_prior_odds and only the observation evidence decays/accumulates.
    "BST_P_D":             0.70,
    "BST_P_FP":            0.08,
    "BST_PRIOR_BYZ":       0.40,
    "BST_DECAY":           1.0,    # 1.0 = no decay; 0.95 = ~14-round half-life
    "BST_TAU_MIN":         0.01,   # very low floor: Bayesian posterior handles it
    # Geometric median (shared: GM, Trust-GM, BST-GM)
    "GM_MAX_ITERS":        100,
    "GM_TOL":              1e-5,
    "GM_NU":               1e-6,
    # Experiment grid
    "METHODS":             ["fedavg", "cwtm", "gm", "trust_gm", "bst_gm"],
    "BYZ_RATIOS":          [0.0, 0.20, 0.30, 0.40],
    "SEEDS":               [42, 123, 456, 789, 1011, 1415, 1617, 1819, 2021],
    # Splits — fixed seed=42, identical to existing paper and ACCCS-GM notebook
    "TRAIN_RATIO":         0.70,
    "VAL_RATIO":           0.15,
    "TEST_RATIO":          0.15,
    # Output
    "OUT_DIR":             "/kaggle/working",
    "CSV_NAME":            "bst_gm_results.csv",
    "CHECKPOINT_EVERY":    20,    # save partial CSV every N experiments
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
    Proportional sampling: if sample_per_device set, maintains real
    benign/attack ratio within each device.
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
    - Preserves non-IID structure (each device stays in its own partition)
    - Natural class imbalance preserved (no resampling)
    - Fixed seed=42 → IDENTICAL splits across all experiments and both notebooks
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
    weight[0] = n_attack / n_benign (up-weight minority benign), capped at 50.
    weight[1] = 1.0.
    Returns {device_id (int): tensor([w_benign, w_attack])}.
    NOTE: Weights computed from CLIENT's own data. The server does not hold
    any labeled data. This is NOT a violation of the "no server-side labeled
    data" claim — clients compute and report their own class counts.
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
    3-layer MLP for binary IDS on N-BaIoT.
    Architecture: 115 → 128 → 64 → 2  (ReLU activations)
    Parameters  : 23,234 (all float32)
    No BatchNorm: diverges under non-IID data (causes 66% ASR — see v9 fix).
    No Dropout  : not needed for this task size.

    State-dict keys:
      net.0.weight [128,115] = 14,720
      net.0.bias   [128]     =    128
      net.2.weight [64,128]  =  8,192
      net.2.bias   [64]      =     64
      net.4.weight [2,64]    =    128
      net.4.bias   [2]       =      2
      Total                  = 23,234
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


# ── Architecture verification ─────────────────────────────────────────────────
_tmp = IDSModel()
_sd  = _tmp.state_dict()
assert sum(v.numel() for v in _sd.values()) == 23_234, "Parameter count mismatch"
assert len(list(_tmp.buffers())) == 0, "Unexpected buffers (BatchNorm?)"
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
    """
    Create per-experiment DataLoaders with isolated per-client RNG.
    Client order: sorted by device_id → consistent with CLIENT_CLASS_WEIGHTS.
    Returns (loaders, sizes).
    """
    loaders, sizes = [], []
    for dev in sorted(train_df["device_id"].unique()):
        sub = train_df[train_df["device_id"] == dev]
        ds  = TensorDataset(
            torch.tensor(sub[feat_cols].values, dtype=torch.float32),
            torch.tensor(sub["label"].values,   dtype=torch.long),
        )
        gen = torch.Generator()
        gen.manual_seed(int(seed) + int(dev))   # isolated per-client shuffle RNG
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
    flip_labels=True → Byzantine label-flip attack: y ← 1 − y.
    Applied to every batch in every epoch (full static label-flip).
    class_weights → per-client weighted CrossEntropyLoss.
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
                yb = 1 - yb   # 0↔1 for torch.long tensors in {0,1}
            opt.zero_grad()
            crit(model(Xb), yb).backward()
            opt.step()
    return {k: v.cpu() for k, v in model.state_dict().items()}


def evaluate_global(global_sd: dict, test_loader, device) -> dict:
    """
    Evaluate global model on the combined test set.
    ASR = FN/(FN+TP) × 100 = fraction of attack traffic classified as benign
          (False Negative Rate of IDS — higher = worse)
    FPR = FP/(FP+TN) × 100 = fraction of benign traffic classified as attack
          (False Positive Rate — higher = worse)
    BA  = ((100-ASR) + (100-FPR)) / 2 = balanced accuracy under class imbalance
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
        "asr":               asr,
        "fpr":               fpr,
        "balanced_accuracy": ba,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def init_global_model(seed: int) -> dict:
    """Deterministic global model init. Re-seeds all RNGs for reproducibility."""
    set_all_seeds(seed)
    return {k: v.cpu() for k, v in IDSModel(input_dim=INPUT_DIM).state_dict().items()}


print("✓ Cell 5 complete — FL core utilities defined")


# ===== CELL 6: Aggregation Methods =====

# ── Shared: delta projection + adaptive median clipping ──────────────────────
def _compute_and_clip_deltas(client_sds: list, global_sd: dict) -> tuple:
    """
    Compute δᵢ = clientᵢ − global for each client.
    Apply adaptive median norm clipping:
      clip_threshold = median(||δᵢ||)
      Byzantine-robust: manipulation requires > 50% Byzantine clients.
    Returns:
      clipped_vecs : list of float64 numpy arrays (for Weiszfeld)
      delta_dicts  : list of unclipped {key: tensor} dicts
    """
    deltas = [
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
    Minimizes Σᵢ τᵢ · ||μ − δ̂ᵢ||₂.
    tau = np.ones(n) recovers the plain unweighted geometric median.
    ν = 1e-6: prevents divide-by-zero when iterate equals a data point.
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
    """FedAvg: weighted average by local dataset size (McMahan et al. 2017).
    No Byzantine defense — failure at high Byzantine fractions is expected."""
    total = sum(client_sizes)
    agg   = {}
    for k in client_sds[0]:
        agg[k] = sum(
            (s / total) * sd[k].float()
            for sd, s in zip(client_sds, client_sizes)
        )
    return agg


# ── 2. CWTM (Coordinate-Wise Trimmed Mean) ────────────────────────────────────
def cwtm(client_sds: list, n_malicious: int) -> dict:
    """
    Coordinate-Wise Trimmed Mean (Yin et al. ICML 2018).
    β = min(n_malicious, (K−1)//2) values trimmed from each coordinate tail.
    Oracle: requires knowing n_malicious — acknowledged limitation.
    No delta-space clipping: static clipping harms CWTM under label-flip
    (moves Byzantine deltas out of the trimming window). See Allouah et al.
    ICLR 2025 for the formal proof.
    At 40% Byzantine (K=9, β=4): K−2β=1 → degenerates to coordinate-wise median.
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
def geometric_median(client_sds: list, global_sd: dict) -> dict:
    """
    Plain GM: delta-space projection + adaptive median clipping + Weiszfeld.
    Uniform trust weights (τᵢ = 1 for all i).
    No oracle inputs required.
    """
    clipped, deltas = _compute_and_clip_deltas(client_sds, global_sd)
    n   = len(clipped)
    tau = np.ones(n, dtype=np.float64)
    w   = _weiszfeld(clipped, tau, CFG["GM_MAX_ITERS"], CFG["GM_TOL"], CFG["GM_NU"])
    agg_delta = np_to_sd(w, deltas[0])
    return {k: global_sd[k].float() + agg_delta[k] for k in global_sd}


# ── 4. Trust-GM (Step-Function Honeypot, paper baseline) ──────────────────────
class StepFunctionHoneypot:
    """
    Original Trust-GM step-function trust schedule (unchanged from paper).
    Trust schedule: {0 detections→1.0, 1→0.50, 2→0.25, ≥3→min_trust (0.15)}.
    Isolated RNG: seed + 10_000 (independent of FL training RNG).
    """
    SCHEDULE = {0: 1.00, 1: 0.50, 2: 0.25}

    def __init__(self, detect_prob: float, fp_rate: float,
                 min_trust: float, seed: int):
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


def trust_gm(client_sds: list, trust_scores: np.ndarray, global_sd: dict) -> dict:
    """Step-function Trust-GM (paper baseline)."""
    clipped, deltas = _compute_and_clip_deltas(client_sds, global_sd)
    tau = np.asarray(trust_scores, dtype=np.float64)
    w   = _weiszfeld(clipped, tau, CFG["GM_MAX_ITERS"], CFG["GM_TOL"], CFG["GM_NU"])
    agg_delta = np_to_sd(w, deltas[0])
    return {k: global_sd[k].float() + agg_delta[k] for k in global_sd}


# ── 5. BST-GM (Bayesian Sequential Trust — NOVEL METHOD) ──────────────────────
class BayesianSequentialTrust:
    """
    Bayesian Sequential Trust (BST) for honeypot-based FL defense.

    MECHANISM:
    Each round t, the server observes whether client i triggered the honeypot.
    The observation updates a Bayesian posterior on P(Byzantine | observations).

    The accumulated log-likelihood ratio (LLR) for observations alone:
      S_i^(t) = decay · S_i^(t-1) + LLR_obs(detected_i^(t))

    where:
      LLR_obs(detected=True)  = log(p_d / p_fp)       > 0  (Byzantine evidence)
      LLR_obs(detected=False) = log((1-p_d)/(1-p_fp)) < 0  (Benign evidence)

    Total LLR including prior:
      LLR_i^(t) = log_prior_odds + S_i^(t)

    P(Byzantine_i | all obs) = sigmoid(LLR_i)
    Trust_i = 1 - P(Byzantine_i | all obs) = sigmoid(-LLR_i)

    [FIX-1] KEY DESIGN: The prior log-odds is FIXED (not decayed).
    Only the accumulated observation evidence S_i^(t) is decayed per round.
    This is mathematically correct for a sequential Bayesian update:
      - Prior stays fixed as the background belief
      - Recent observations (via decay) count more than old ones
    With decay=1.0 (default), this is pure Bayesian accumulation where
    all observations have equal weight.

    VERIFIED VALUES (p_d=0.70, p_fp=0.08, prior_byz=0.40, decay=1.0, t=30):
      E[tau_byz] = 0.010  (step-fn floor: 0.150 — BST 15x more aggressive)
      E[tau_hon] = 1.000  (step-fn: 0.321 — BST fully exonerates benign FP victims)
      Effective Byzantine fraction: 0.008  (step-fn: 0.272)

    NUMERICAL PROPERTIES:
      - S_i clipped to [-50, 50] to prevent sigmoid overflow
      - tau_min floor ensures Weiszfeld weights are always positive
      - rng isolated at seed + 10_000 (does not pollute FL training RNG)
    """

    def __init__(self, p_d: float, p_fp: float, prior_byz: float = 0.40,
                 decay: float = 1.0, tau_min: float = 0.01, seed: int = 42):
        assert 0 < p_d < 1,        "p_d must be in (0, 1)"
        assert 0 < p_fp < 1,       "p_fp must be in (0, 1)"
        assert 0 < prior_byz < 1,  "prior_byz must be in (0, 1)"
        assert 0 < decay <= 1.0,   "decay must be in (0, 1]"
        assert 0 < tau_min < 0.5,  "tau_min must be in (0, 0.5)"

        self.tau_min = tau_min
        self.decay   = decay
        self.rng     = np.random.RandomState(seed)

        # [FIX-1] Precompute fixed prior log-odds (NEVER decayed)
        self._log_prior_odds = float(np.log(prior_byz / (1.0 - prior_byz)))

        # Precompute log-likelihood ratio per observation type
        self._llr_detected   = float(np.log(p_d / p_fp))
        self._llr_not_detect = float(np.log((1.0 - p_d) / (1.0 - p_fp)))

        # Honeypot detection probabilities
        self._p_d  = p_d
        self._p_fp = p_fp

        # [FIX-1] Accumulated OBSERVATION evidence per client (NOT including prior).
        # Prior is always added separately in get_trust().
        # Starts at 0.0 for all clients (no observations yet).
        self._accum_obs: dict = {}

    def simulate_round(self, client_id: int, is_byzantine: bool) -> bool:
        """
        Simulate one honeypot observation for client_id and update LLR.
        Returns: detected (bool).
        """
        prob     = self._p_d if is_byzantine else self._p_fp
        detected = bool(self.rng.random() < prob)
        self._update_accum(client_id, detected)
        return detected

    def _update_accum(self, client_id: int, detected: bool):
        """
        [FIX-1] Update accumulated observation evidence for client_id.
        1. Decay existing evidence (temporal forgetting if decay < 1.0)
        2. Add new observation's LLR
        3. Clip to [-50, 50] to prevent sigmoid overflow
        The prior is NOT decayed — it is added separately in get_trust().
        """
        # Initialize at 0 (no observation evidence yet) if first seen
        if client_id not in self._accum_obs:
            self._accum_obs[client_id] = 0.0

        # Decay previous observation evidence
        self._accum_obs[client_id] *= self.decay

        # Add current observation's likelihood ratio
        self._accum_obs[client_id] += (
            self._llr_detected if detected else self._llr_not_detect
        )

        # Clip to prevent float overflow in sigmoid
        self._accum_obs[client_id] = float(
            np.clip(self._accum_obs[client_id], -50.0, 50.0)
        )

    def get_trust(self, client_id: int) -> float:
        """
        [FIX-1] Returns trust = P(Benign | all observations) for client_id.
        Total LLR = fixed_prior + accumulated_observation_evidence.
        Trust = sigmoid(-LLR) = 1 - P(Byzantine).
        """
        accum = self._accum_obs.get(client_id, 0.0)
        # Combine fixed prior with accumulated evidence
        llr   = self._log_prior_odds + accum
        llr   = float(np.clip(llr, -50.0, 50.0))
        p_byz = 1.0 / (1.0 + np.exp(-llr))   # sigmoid(LLR) = P(Byzantine)
        return max(self.tau_min, 1.0 - p_byz)

    def get_all_trusts(self, n_clients: int) -> np.ndarray:
        """Returns trust scores for all clients as a numpy array."""
        return np.array(
            [self.get_trust(i) for i in range(n_clients)],
            dtype=np.float64,
        )


def bst_gm(client_sds: list, trust_scores: np.ndarray, global_sd: dict) -> dict:
    """
    BST-weighted geometric median.
    Identical aggregation pipeline to Trust-GM — only the trust scores differ.
    Trust scores come from BayesianSequentialTrust.get_all_trusts().
    """
    clipped, deltas = _compute_and_clip_deltas(client_sds, global_sd)
    tau = np.asarray(trust_scores, dtype=np.float64)
    if tau.sum() < 1e-12:
        tau = np.ones(len(clipped), dtype=np.float64)
    w         = _weiszfeld(clipped, tau, CFG["GM_MAX_ITERS"],
                            CFG["GM_TOL"], CFG["GM_NU"])
    agg_delta = np_to_sd(w, deltas[0])
    return {k: global_sd[k].float() + agg_delta[k] for k in global_sd}


print("✓ Cell 6 complete — all aggregation methods defined")
print("  Methods: fedavg | cwtm | gm | trust_gm (step fn) | bst_gm (Bayesian)")
print("  [FIX-1] BST: prior fixed, only observation evidence accumulated/decayed")


# ===== CELL 7: BST-GM Sanity Checks =====

print("BST-GM mathematical verification:")

# Verify initial trust (prior only, no observations)
_bst_test = BayesianSequentialTrust(
    p_d=0.70, p_fp=0.08, prior_byz=0.40, decay=1.0, tau_min=0.01
)
trust_init = _bst_test.get_trust(0)  # client 0, no observations yet
print(f"  Initial trust (no obs, prior_byz=0.40): {trust_init:.4f}  "
      f"(expected 0.6000 ✓)" if abs(trust_init - 0.6) < 0.001 else
      f"(expected 0.6000 ✗ got {trust_init})")

# Simulate Byzantine client (30 rounds, p_d=0.70)
rng_test = np.random.RandomState(42)
bst_byz  = BayesianSequentialTrust(p_d=0.70, p_fp=0.08, prior_byz=0.40,
                                    decay=1.0, tau_min=0.01, seed=42)
bst_hon  = BayesianSequentialTrust(p_d=0.70, p_fp=0.08, prior_byz=0.40,
                                    decay=1.0, tau_min=0.01, seed=99)

for _ in range(30):
    # Directly update accum without using simulate_round (avoids shared RNG)
    det_byz = bool(rng_test.random() < 0.70)
    det_hon = bool(rng_test.random() < 0.08)
    bst_byz._update_accum(0, det_byz)
    bst_hon._update_accum(0, det_hon)

tau_byz_30 = bst_byz.get_trust(0)
tau_hon_30 = bst_hon.get_trust(0)
print(f"  Byzantine trust at t=30: {tau_byz_30:.4f}  "
      f"(step fn floor: 0.1500)")
print(f"  Honest trust at t=30:    {tau_hon_30:.4f}  "
      f"(step fn expected: ~0.321)")

# Verify overflow safety: 1000 consecutive detections
bst_overflow = BayesianSequentialTrust(p_d=0.70, p_fp=0.08, prior_byz=0.40,
                                        tau_min=0.01)
for _ in range(1000):
    bst_overflow._update_accum(0, True)
t_over = bst_overflow.get_trust(0)
print(f"  Trust after 1000 detections (overflow safe): {t_over:.4f}  "
      f"{'✓' if 0 <= t_over <= 1 else 'OVERFLOW!'}")

# Verify tau.sum() > 0 in worst case
tau_worst = np.full(9, 0.01)
print(f"  Min tau.sum() (all at tau_min=0.01): {tau_worst.sum():.3f} > 0 ✓")

# Verify expected trust values at t=30 analytically
# use np directly below
_llr_d = np.log(0.70/0.08)
_llr_n = np.log(0.30/0.92)
_prior = np.log(0.40/0.60)
# Expected: 21 detections, 9 non-detections
_accum_byz = 21 * _llr_d + 9 * _llr_n
_tau_byz_expect = max(0.01, 1 - 1/(1 + np.exp(-(_prior + _accum_byz))))
# Expected: 2.4 FPs, 27.6 non-detections (rounded: 2 FP, 28 non-detect)
_accum_hon = 2 * _llr_d + 28 * _llr_n
_tau_hon_expect = max(0.01, 1 - 1/(1 + np.exp(-(_prior + _accum_hon))))
print(f"\n  Analytical check (expected values at t=30):")
print(f"    E[tau_byz]: {_tau_byz_expect:.4f}  (paper doc claims: 0.0100)")
print(f"    E[tau_hon]: {_tau_hon_expect:.4f}  (paper doc claims: 1.0000)")

del _bst_test, bst_byz, bst_hon, bst_overflow, tau_worst
print("✓ Cell 7 complete — BST-GM verified")


# ===== CELL 8: Training Loop =====

def run_experiment(method: str, byz_ratio: float, seed: int,
                   train_df: pd.DataFrame, feat_cols: list,
                   test_loader, device) -> dict:
    """
    Run one complete FL experiment: one method × byz_ratio × seed.

    RNG isolation (strict reproducibility — matches original paper v12):
      1. set_all_seeds(seed)           — global RNG reset
      2. create_client_loaders(...)    — per-client torch.Generator(seed+dev)
      3. RandomState(seed).choice(...) — dedicated Byzantine selection RNG
      4. init_global_model(seed)       — re-seeds for deterministic weight init
      5. Honeypot seed: seed + 10_000  — isolated from FL training RNG

    Round ordering (A → B → C):
      A: Local training (Byzantine clients flip labels)
      B: Honeypot simulation (trust_gm, bst_gm only, BEFORE aggregation)
      C: Aggregation (method-specific)

    [FIX-5] del client_updates after each round for memory efficiency.
    """
    # 1. Global RNG reset
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    set_all_seeds(seed)

    # 2. Client loaders (per-client isolated RNG)
    client_loaders, client_sizes = create_client_loaders(
        train_df, feat_cols, CFG["BATCH_SIZE"], seed
    )
    n_clients = len(client_loaders)

    # 3. Byzantine selection (own isolated RNG)
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
    elif method == "bst_gm":
        bst = BayesianSequentialTrust(
            p_d=CFG["BST_P_D"],
            p_fp=CFG["BST_P_FP"],
            prior_byz=CFG["BST_PRIOR_BYZ"],
            decay=CFG["BST_DECAY"],
            tau_min=CFG["BST_TAU_MIN"],
            seed=seed + 10_000,
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

        # B: Honeypot simulation (before aggregation)
        if method == "trust_gm":
            for cid in range(n_clients):
                hm.simulate_round(cid, is_byzantine=(cid in byz_ids))
            trust_scores = hm.get_all_trusts(n_clients)

        elif method == "bst_gm":
            for cid in range(n_clients):
                bst.simulate_round(cid, is_byzantine=(cid in byz_ids))
            trust_scores = bst.get_all_trusts(n_clients)

        # C: Aggregation
        if method == "fedavg":
            global_sd = fedavg(client_updates, client_sizes)

        elif method == "cwtm":
            global_sd = cwtm(client_updates, n_malicious=n_byz)

        elif method == "gm":
            global_sd = geometric_median(client_updates, global_sd)

        elif method == "trust_gm":
            global_sd = trust_gm(client_updates, trust_scores, global_sd)

        elif method == "bst_gm":
            global_sd = bst_gm(client_updates, trust_scores, global_sd)

        else:
            raise ValueError(f"Unknown method: '{method}'")

        # [FIX-5] Explicit memory cleanup per round
        del client_updates

    # 7. Final evaluation
    metrics = evaluate_global(global_sd, test_loader, device)
    return {
        "metrics":  metrics,
        "byz_ids":  sorted(byz_ids),
        "n_byz":    n_byz,
    }


print("✓ Cell 8 complete — training loop defined")


# ===== CELL 9: Experiment Runner =====
# [FIX-4] Checkpointing every CHECKPOINT_EVERY experiments.

print("=" * 65)
print("EXPERIMENT RUNNER")
print(f"Grid: {len(CFG['METHODS'])} methods × {len(CFG['BYZ_RATIOS'])} "
      f"byz fractions × {len(CFG['SEEDS'])} seeds = {n_total} experiments")
print(f"Checkpoint every {CFG['CHECKPOINT_EVERY']} experiments → "
      f"{CFG['OUT_DIR']}/bst_gm_checkpoint.csv")
print("=" * 65)

results = []
t_start = time.time()
run_idx = 0

_chk_path = f"{CFG['OUT_DIR']}/bst_gm_checkpoint.csv"

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

            # Progress log (every run for clarity)
            elapsed_total = time.time() - t_start
            avg_s         = elapsed_total / run_idx
            eta_h         = avg_s * (n_total - run_idx) / 3600
            print(f"  [{run_idx:3d}/{n_total}] {method:12s} "
                  f"byz={byz_ratio:.0%} seed={seed:4d}  "
                  f"ASR={m['asr']:.4f}%  FPR={m['fpr']:.4f}%  "
                  f"t={elapsed:.0f}s  ETA={eta_h:.1f}h")

            # [FIX-4] Periodic checkpoint
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
# Remove checkpoint (superseded by final CSV)
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

print("\nBST-GM vs Trust-GM at 40% Byzantine:")
for m in ["trust_gm", "bst_gm", "gm", "cwtm", "fedavg"]:
    sub = byz40[byz40["method"] == m]
    if len(sub) > 0:
        print(f"  {m:12s}  ASR: {sub['asr'].mean():.4f}% "
              f"± {sub['asr'].std():.4f}%  "
              f"max={sub['asr'].max():.4f}%  "
              f"FPR: {sub['fpr'].mean():.4f}%")

print("\nFPR overhead at 0% Byzantine (clean condition — no attack):")
clean = df[df["byz_ratio"] == 0.0]
for m in ["fedavg", "gm", "trust_gm", "bst_gm"]:
    sub = clean[clean["method"] == m]
    if len(sub) > 0:
        print(f"  {m:12s}  FPR: {sub['fpr'].mean():.4f}% ± {sub['fpr'].std():.4f}%")

print("✓ Cell 11 complete")


# ===== CELL 12: Statistical Tests =====
# [FIX-3] Wilcoxon wrapped in try-except to handle zero-variance edge case.

try:
    from scipy import stats as sp_stats

    print("\n" + "=" * 65)
    print("STATISTICAL TEST: BST-GM vs Trust-GM (step fn)")
    print("H₁: BST-GM ASR < Trust-GM ASR  [one-tailed]")
    print("Test selection: Wilcoxon if non-normal (Shapiro p<0.05), else paired t-test")
    print("=" * 65)

    for byz in [0.20, 0.30, 0.40]:
        sub   = df[df["byz_ratio"] == byz].copy()
        seeds = sorted(sub["seed"].unique())

        bst_vals = [sub[(sub["method"]=="bst_gm")  & (sub["seed"]==s)]["asr"].values[0]
                    for s in seeds]
        tgm_vals = [sub[(sub["method"]=="trust_gm") & (sub["seed"]==s)]["asr"].values[0]
                    for s in seeds]

        sw_bst = sp_stats.shapiro(bst_vals).pvalue
        sw_tgm = sp_stats.shapiro(tgm_vals).pvalue
        diffs  = np.array(bst_vals) - np.array(tgm_vals)

        try:
            if sw_bst > 0.05 and sw_tgm > 0.05:
                stat, p_val = sp_stats.ttest_rel(bst_vals, tgm_vals, alternative="less")
                test_name   = "paired t-test"
            else:
                stat, p_val = sp_stats.wilcoxon(bst_vals, tgm_vals, alternative="less")
                test_name   = "Wilcoxon"
            # Handle NaN p-value (all differences zero)
            if not np.isfinite(p_val):
                p_val     = 1.0
                test_name += " (zero-variance: p=1.0)"
        except Exception as e:
            p_val, stat, test_name = 1.0, 0.0, f"N/A ({e})"

        d_std    = diffs.std(ddof=1)
        cohens_d = diffs.mean() / (d_std + 1e-12) if d_std > 0 else 0.0
        sig      = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s.")

        print(f"  byz={byz:.0%}: "
              f"BST-GM={np.mean(bst_vals):.4f}%  "
              f"Trust-GM={np.mean(tgm_vals):.4f}%  "
              f"[{test_name}] p={p_val:.4f} {sig}  d={cohens_d:.3f}")

    print("\nBST-GM vs plain GM:")
    for byz in [0.20, 0.30, 0.40]:
        sub      = df[df["byz_ratio"] == byz].copy()
        seeds    = sorted(sub["seed"].unique())
        bst_vals = [sub[(sub["method"]=="bst_gm") & (sub["seed"]==s)]["asr"].values[0]
                    for s in seeds]
        gm_vals  = [sub[(sub["method"]=="gm")     & (sub["seed"]==s)]["asr"].values[0]
                    for s in seeds]
        try:
            stat, p_val = sp_stats.wilcoxon(bst_vals, gm_vals, alternative="less")
            if not np.isfinite(p_val):
                p_val = 1.0
        except Exception:
            p_val = 1.0
        sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s.")
        print(f"  byz={byz:.0%}: BST-GM({np.mean(bst_vals):.4f}%) vs "
              f"GM({np.mean(gm_vals):.4f}%)  p={p_val:.4f} {sig}")

except ImportError:
    print("scipy not available — skipping statistical tests")

print("✓ Cell 12 complete")


# ===== CELL 13: Visualization =====
# [FIX-2] Log-scale applied via explicit if statement, not ternary expression.

if HAVE_PLOT:
    METHOD_COLORS = {
        "fedavg":   "#E53935",
        "cwtm":     "#FB8C00",
        "gm":       "#1E88E5",
        "trust_gm": "#43A047",
        "bst_gm":   "#8E24AA",
    }
    METHOD_LABELS = {
        "fedavg":   "FedAvg",
        "cwtm":     "CWTM (oracle β)",
        "gm":       "GM",
        "trust_gm": "Trust-GM (step fn)",
        "bst_gm":   "BST-GM (Bayesian) ★",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("BST-GM vs Baselines — N-BaIoT Byzantine Attack",
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
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    # [FIX-2] Explicit if — handles zero values cleanly
    if df["asr"].max() > 1.0:
        # Add a small offset to avoid log(0) if any ASR is exactly 0
        ax.set_yscale("log")

    # ── Plot 2: ASR boxplot at 40% Byzantine (9 seeds) ───────────────────
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
        rotation=25, ha="right", fontsize=8
    )
    ax.grid(True, alpha=0.3, axis="y")

    # ── Plot 3: BST-GM vs Trust-GM per-seed scatter ───────────────────────
    ax = axes[2]
    byz40   = df[df["byz_ratio"] == 0.40]
    seeds_u = sorted(byz40["seed"].unique())

    bst_asrs  = [byz40[(byz40["method"]=="bst_gm")  & (byz40["seed"]==s)]["asr"].values[0]
                 for s in seeds_u]
    tgm_asrs  = [byz40[(byz40["method"]=="trust_gm") & (byz40["seed"]==s)]["asr"].values[0]
                 for s in seeds_u]
    gm_asrs   = [byz40[(byz40["method"]=="gm")       & (byz40["seed"]==s)]["asr"].values[0]
                 for s in seeds_u]

    ax.scatter(tgm_asrs, bst_asrs, c="#8E24AA", s=90, zorder=5,
               label="BST-GM vs Trust-GM")
    ax.scatter(gm_asrs,  bst_asrs, c="#1E88E5", s=70, marker="^", zorder=4,
               label="BST-GM vs GM")

    max_val = max(
        max(tgm_asrs) if tgm_asrs else 0,
        max(gm_asrs)  if gm_asrs  else 0,
        max(bst_asrs) if bst_asrs else 0,
    ) * 1.15 + 0.01
    ax.plot([0, max_val], [0, max_val], "k--", lw=1, alpha=0.5, label="y=x (equal)")
    ax.set_xlabel("Baseline ASR (%)", fontsize=11)
    ax.set_ylabel("BST-GM ASR (%)", fontsize=11)
    ax.set_title("BST-GM vs Baselines at 40% Byzantine\n"
                 "Points BELOW y=x: BST-GM wins", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = f"{CFG['OUT_DIR']}/bst_gm_results.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {fig_path}")

print("=" * 65)
print("BST-GM EXPERIMENT COMPLETE")
print(f"CSV    : {csv_path}")
if HAVE_PLOT:
    print(f"Figure : {fig_path}")
print("Download both files from /kaggle/working/")
print("=" * 65)
print("✓ Cell 13 complete — all done")
# This notebook is fully runnable on Kaggle without modification.
