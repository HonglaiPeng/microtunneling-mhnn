import os
import random
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler


# =========================================================
# 0. notebook name
# =========================================================
MODEL_VERSION = "Stage-Wise MHNN Joint-NN Comparison"
NOTEBOOK_NAME = MODEL_VERSION
print(NOTEBOOK_NAME)



# =========================================================
# 1. reproducibility
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------------------------------------------------
# reproducible wall-clock timing
# ---------------------------------------------------------
def synchronize_cuda_for_timing():
    """Synchronize CUDA so GPU work is included in elapsed-time measurements."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_now():
    """Return a high-resolution, monotonic timestamp."""
    synchronize_cuda_for_timing()
    return time.perf_counter()



# =========================================================
# 2. user settings and physical constants
# =========================================================


START_DIST = 80.0

DATA_COVERAGE_MODE = "lub_only"

if DATA_COVERAGE_MODE not in {"full_drive", "lub_only"}:
    raise ValueError(
        "DATA_COVERAGE_MODE must be either 'full_drive' or 'lub_only'."
    )

DATASET_SPECS = [
    "Phase1_BayesianAligned_V2.5_02_800Drives_StartLubAt80_01.xlsx",
]

sheet_name = 0


# ---------------------------------------------------------
# sequence and batch-size settings
# ---------------------------------------------------------
WINDOW_SIZE = 50
WINDOW_SIZES = [50]
window_size = WINDOW_SIZE
WINDOW_TAG = f"window_{WINDOW_SIZE}"

BATCH_SIZE = 512
BATCH_SIZES = [512]
batch_size = BATCH_SIZE
BATCH_TAG = f"batch_{BATCH_SIZE}"


# ---------------------------------------------------------
# staged training
# ---------------------------------------------------------
# Stage 1: learn N0
# Stage 2: learn latent lubrication
# Stage 3: learn mu
# Stage 4: learn phi
# Stage 5: learn gamma
TRAINING_STAGE = 5


# ---------------------------------------------------------
# conditional-head teacher-forcing placeholder
# ---------------------------------------------------------
CONDITIONAL_PLACEHOLDER_MODE = "predicted"
CONDITIONAL_PLACEHOLDER_STAGES = (3, 4, 5)


# ---------------------------------------------------------
# assessment loss switches
# ---------------------------------------------------------


USE_PARAMETER_SUPERVISION_IN_GRADIENT = False
USE_LUBRICATION_SUPERVISION_IN_GRADIENT = False
USE_PHYSICS_INFORMED_LOSS_IN_GRADIENT = True

# Basic force losses remain active independently.
USE_BASIC_FACE_FORCE_IN_GRADIENT = True
USE_BASIC_FRICTION_FORCE_IN_GRADIENT = True
USE_BASIC_TOTAL_FORCE_IN_GRADIENT = True


REPORT_LUBRICATION_LOSSES = True


high_level_gradient_enabled = any([
    USE_PARAMETER_SUPERVISION_IN_GRADIENT,
    USE_LUBRICATION_SUPERVISION_IN_GRADIENT,
    USE_PHYSICS_INFORMED_LOSS_IN_GRADIENT,
])

basic_force_gradient_enabled = any([
    USE_BASIC_FACE_FORCE_IN_GRADIENT,
    USE_BASIC_FRICTION_FORCE_IN_GRADIENT,
    USE_BASIC_TOTAL_FORCE_IN_GRADIENT,
])

if not (
    high_level_gradient_enabled
    or basic_force_gradient_enabled
):
    raise ValueError(
        "At least one gradient path must be enabled: "
        "parameter supervision, lubrication supervision, "
        "physics-informed loss, or basic force reconstruction."
    )


def assessment_switch_tag(enabled):
    return "on" if enabled else "off"


conditional_tag = (
    "teacher_forced"
    if CONDITIONAL_PLACEHOLDER_MODE == "true"
    else "predicted_conditional"
)


# ---------------------------------------------------------
# experiment output directory
# ---------------------------------------------------------
save_dir = (
    "MHNN"
)

os.makedirs(save_dir, exist_ok=True)



PLACEHOLDER_MODE = "fixed"


# ---------------------------------------------------------
# geometry
# ---------------------------------------------------------
pipe_diameter = 1.524
tbm_diameter = 1.5748


# ---------------------------------------------------------
# fixed reference values
# ---------------------------------------------------------
REF_N0 = 2.5
REF_MU = 0.58
REF_PHI_DEG = 32.0
REF_GAMMA = 17.28


# ---------------------------------------------------------
# input, parameter, latent, and physics targets
# ---------------------------------------------------------
INPUT_COLS = [
    "Jacking Forces",
    "Jacking Distance",
    "F_face",
    "True_F_fric_lub",
]

INPUT_IDX = {
    column_name: index
    for index, column_name in enumerate(INPUT_COLS)
}

PARAM_COLS = [
    "N0",
    "mu",
    "phi_deg",
    "gamma",
]

BASE_COLS = [
    "Base_phi_deg",
    "Base_gamma",
]

LUB_COLS = [
    "Actual_F_lub",
]

PHYSICS_COLS = [
    "F_face",
    "K_fric_local",
    "F_fric_unlub",
    "True_F_total_lub",
]

AUX_COLS = [
    "Local_Distance_From_Start",
    "F_fric_at_start",
]


# =========================================================
# sequence, batch-size, and training settings
# =========================================================

ARCHITECTURE_NAME = "MHNN"

WINDOW_SIZES_TO_COMPARE = WINDOW_SIZES
BATCH_SIZES_TO_COMPARE = BATCH_SIZES

if WINDOW_SIZE not in WINDOW_SIZES_TO_COMPARE:
    raise ValueError(
        f"WINDOW_SIZE={WINDOW_SIZE} is not included in "
        f"WINDOW_SIZES_TO_COMPARE={WINDOW_SIZES_TO_COMPARE}"
    )

if BATCH_SIZE not in BATCH_SIZES_TO_COMPARE:
    raise ValueError(
        f"BATCH_SIZE={BATCH_SIZE} is not included in "
        f"BATCH_SIZES_TO_COMPARE={BATCH_SIZES_TO_COMPARE}"
    )


# ---------------------------------------------------------
# common target alignment
# ---------------------------------------------------------
MAX_WINDOW_SIZE = max(WINDOW_SIZES_TO_COMPARE)
COMMON_TARGET_START_POS = MAX_WINDOW_SIZE - 1


# ---------------------------------------------------------
# model and training settings
# ---------------------------------------------------------
dropout = 0.10
num_epochs = 50
PATIENCE = 80
SCHEDULER_PATIENCE = 20

train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

weight_decay = 1e-4


# ---------------------------------------------------------
# global loss weights
# ---------------------------------------------------------
lambda_param = 1.0
lambda_phys_direct = 0.10
lambda_consistency = 0.01
lambda_face_target = 0.50

lambda_total_force = 0.20
lambda_friction_obs = 0.20

lambda_bound = 0.01
lambda_residual_prior = 0.01

lambda_lub_supervised = 0.50
lambda_lub_prior = 0.01
lambda_lub_raw_force = 0.50

LUB_EPS = 1e-3


# ---------------------------------------------------------
# physical bounds
# ---------------------------------------------------------
BOUNDS = {
    "N0": (2.2, 2.9),
    "mu": (0.40, 0.80),
    "phi_deg": (31.0, 33.0),
    "gamma": (16.0, 18.0),
}


# ---------------------------------------------------------
# dynamic residual settings
# ---------------------------------------------------------
DRIVE_BOUND_CV_PHI = 0.075
DRIVE_BOUND_CV_GAMMA = 0.05


# ---------------------------------------------------------
# reporting tolerance structure
# ---------------------------------------------------------
PHI_TOL_DEG = 0.75
GAMMA_TOL = 0.75

PHI_TOL_LIST = [0.5, 0.75, 1.0, 1.5]
GAMMA_TOL_LIST = [0.25, 0.5, 0.75, 1.0]


def tol_key(prefix, tol):
    return f"{prefix}_within_tol_{str(tol).replace('.', 'p')}"


TOLERANCE_METRIC_KEYS = (
    [
        tol_key("phi", tol)
        for tol in PHI_TOL_LIST
    ]
    + [
        tol_key("gamma", tol)
        for tol in GAMMA_TOL_LIST
    ]
)


# =========================================================
# configuration printout
# =========================================================

print("\n" + "=" * 90)
print("MHNN EXPERIMENT CONFIGURATION")
print("=" * 90)

print(f"Architecture:              {ARCHITECTURE_NAME}")
print(f"Training stage:             {TRAINING_STAGE}")
print(f"Dataset coverage:           {DATA_COVERAGE_MODE}")
print(f"Window size:                {WINDOW_SIZE}")
print(f"Batch size:                 {BATCH_SIZE}")
print(f"Conditional mode:           {CONDITIONAL_PLACEHOLDER_MODE}")
print(f"Placeholder mode:           {PLACEHOLDER_MODE}")
print(f"Save directory:             {save_dir}")

print("\nHigh-level gradient switches:")
print(
    "  Parameter supervision:    "
    f"{USE_PARAMETER_SUPERVISION_IN_GRADIENT}"
)
print(
    "  Lubrication supervision:  "
    f"{USE_LUBRICATION_SUPERVISION_IN_GRADIENT}"
)
print(
    "  Physics-informed loss:    "
    f"{USE_PHYSICS_INFORMED_LOSS_IN_GRADIENT}"
)

print("\nBasic force-loss gradient switches:")
print(
    "  Face resistance:          "
    f"{USE_BASIC_FACE_FORCE_IN_GRADIENT}"
)
print(
    "  Friction force:           "
    f"{USE_BASIC_FRICTION_FORCE_IN_GRADIENT}"
)
print(
    "  Total jacking force:      "
    f"{USE_BASIC_TOTAL_FORCE_IN_GRADIENT}"
)

print("\nDynamic bounded residual settings:")
print("  phi head:                 dynamic base + bounded residual")
print("  gamma head:               dynamic base + bounded residual")
print("  phi/gamma soft bounds:    enabled using physical bounds")
print("  residual references:      Base_phi_deg and Base_gamma")
print(f"  phi drive-bound CV:       {DRIVE_BOUND_CV_PHI:.6f}")
print(f"  gamma drive-bound CV:     {DRIVE_BOUND_CV_GAMMA:.6f}")
print("  global bounds:            safety bounds")

print("\nPhysical references:")
print(f"  REF_N0:                   {REF_N0:.6f}")
print(f"  REF_MU:                   {REF_MU:.6f}")
print(f"  REF_PHI_DEG:              {REF_PHI_DEG:.6f}")
print(f"  REF_GAMMA:                {REF_GAMMA:.6f}")

print("=" * 90)



# =========================================================
# 3. load dataset
# =========================================================
def compute_f_face_from_N0(df_in):
    f_face = 10.0 * 1.32 * np.pi * tbm_diameter * df_in["N0"].values
    F_face = f_face * np.pi * (tbm_diameter / 2.0) ** 2
    return F_face


def lognormal_params(mean, cov):
    sigma = np.sqrt(np.log(1.0 + cov ** 2))
    mu = np.log(mean) - 0.5 * sigma ** 2
    return mu, sigma


def estimate_lubrication_prior_from_residuals(df_in):


    phi_ref_rad = np.radians(REF_PHI_DEG)

    f_fric_unit_ref = (
        REF_MU
        * REF_GAMMA
        * pipe_diameter
        * np.cos(np.pi / 4.0 + phi_ref_rad / 2.0)
        / (2.0 * np.tan(phi_ref_rad))
    )

    K_fric_ref = f_fric_unit_ref * np.pi * pipe_diameter

    F_fric_no_lub_ref = K_fric_ref * df_in["Jacking Distance"].values

    # Version 4.1.10:
    # Use the generator-provided effective lubricated friction.
    # Do NOT use F_fric_unlub here.
    F_fric_observed = df_in["True_F_fric_lub"].values

    F_lub_residual = F_fric_no_lub_ref - F_fric_observed
    F_lub_residual = np.clip(F_lub_residual, a_min=LUB_EPS, a_max=None)

    mean_flub = F_lub_residual.mean()
    std_flub = F_lub_residual.std(ddof=1)
    cov_flub = std_flub / (mean_flub + 1e-12)

    mu_log_flub, sigma_log_flub = lognormal_params(mean_flub, cov_flub)

    return {
        "F_lub_residual": F_lub_residual,
        "mean_flub": mean_flub,
        "std_flub": std_flub,
        "cov_flub": cov_flub,
        "mu_log_flub": mu_log_flub,
        "sigma_log_flub": sigma_log_flub,
    }


def load_one_dataset(file_path, source_name):
    if file_path.endswith(".csv"):
        df_in = pd.read_csv(file_path)
    else:
        df_in = pd.read_excel(file_path, sheet_name=sheet_name)

    df_in = df_in.loc[:, ~df_in.columns.duplicated()].copy()

    if "phi" in df_in.columns and "phi_deg" not in df_in.columns:
        df_in.rename(columns={"phi": "phi_deg"}, inplace=True)

        if df_in["phi_deg"].dropna().median() < 5.0:
            df_in["phi_deg"] = df_in["phi_deg"] * 180.0 / np.pi


    if "Base_phi_deg" not in df_in.columns:
        df_in["Base_phi_deg"] = REF_PHI_DEG
    if "Base_gamma" not in df_in.columns:
        df_in["Base_gamma"] = REF_GAMMA

    if "True_F_total_lub" not in df_in.columns and "True_F_total" in df_in.columns:
        df_in.rename(columns={"True_F_total": "True_F_total_lub"}, inplace=True)

    if "Local_Distance_From_Start" not in df_in.columns:
        df_in["Local_Distance_From_Start"] = np.maximum(
            df_in["Jacking Distance"] - START_DIST,
            0.0,
        )

    if "F_face" not in df_in.columns:
        if "N0" not in df_in.columns:
            raise ValueError(
                f"F_face is missing and N0 is not available in {file_path}."
            )
        df_in["F_face"] = compute_f_face_from_N0(df_in)
        print(f"F_face was missing and was computed from N0 in {file_path}.")

    if "Drive_ID" not in df_in.columns:
        raise ValueError(f"Drive_ID is missing from {file_path}.")

    original_drive_ids = pd.to_numeric(
        df_in["Drive_ID"],
        errors="raise",
    ).astype(int)

    df_in["Dataset_Source"] = source_name
    df_in["Original_Drive_ID"] = original_drive_ids

    # Drive_ID must be unique for window construction.
    df_in["Drive_ID"] = (
        source_name
        + "_drive_"
        + original_drive_ids.astype(str)
    )


    df_in["Split_Group_ID"] = "drive_" + original_drive_ids.astype(str)

    return df_in


dataset_frames = []

for dataset_path in DATASET_SPECS:
    dataset_source = os.path.basename(dataset_path)
    print(f"\nLoading dataset: {dataset_source}")
    print(f"Path: {dataset_path}")
    dataset_frames.append(
        load_one_dataset(
            file_path=dataset_path,
            source_name=dataset_source,
        )
    )

df = pd.concat(dataset_frames, ignore_index=True)


if "True_F_fric_lub" not in df.columns:
    raise ValueError(
        "True_F_fric_lub is missing from the dataset. "
        "For Version 4.1.10, use the generated effective lubricated friction "
        "column True_F_fric_lub as the friction input."
    )


friction_check_error = (
    df["True_F_fric_lub"].values
    - (df["Jacking Forces"].values - df["F_face"].values)
)

print("\nTrue_F_fric_lub consistency check:")
print("  Compare True_F_fric_lub against Jacking Forces - F_face")
print(f"  mean difference:          {np.mean(friction_check_error):.6f}")
print(f"  mean absolute difference: {np.mean(np.abs(friction_check_error)):.6f}")
print(f"  max absolute difference:  {np.max(np.abs(friction_check_error)):.6f}")

required_cols = list(
    dict.fromkeys(
        [
            "Drive_ID",
            "Dataset_Source",
            "Original_Drive_ID",
            "Split_Group_ID",
        ]
        + INPUT_COLS
        + PARAM_COLS
        + BASE_COLS
        + LUB_COLS
        + PHYSICS_COLS
        + AUX_COLS
    )
)

missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns:\n{missing_cols}")

df = df[required_cols].dropna().reset_index(drop=True)


base_variation = df.groupby("Drive_ID", sort=False)[BASE_COLS].nunique()
inconsistent_base_drives = base_variation.index[
    (base_variation > 1).any(axis=1)
].tolist()

if inconsistent_base_drives:
    raise ValueError(
        "Base_phi_deg and Base_gamma must be constant within each Drive_ID. "
        f"Inconsistent drives: {inconsistent_base_drives[:10]}"
    )

df = (
    df
    .sort_values(["Original_Drive_ID", "Dataset_Source", "Jacking Distance"])
    .reset_index(drop=True)
)

if DATA_COVERAGE_MODE == "lub_only":
    df = df[
        df["Jacking Distance"] >= START_DIST - 1e-9
    ].reset_index(drop=True)

if df.empty:
    raise ValueError("No rows remain after applying DATA_COVERAGE_MODE.")

print(f"\nData coverage mode: {DATA_COVERAGE_MODE}")
print(f"Distance range: {df['Jacking Distance'].min():.1f} to {df['Jacking Distance'].max():.1f} m")

print("Loaded dataset shape:", df.shape)
print("Number of drives:", df["Drive_ID"].nunique())
print("Number of split groups:", df["Split_Group_ID"].nunique())
print("Dataset sources:")
print(df["Dataset_Source"].value_counts())

print("\nInput columns:")
print(INPUT_COLS)

print("\nLatent lubrication target columns:")
print(LUB_COLS)

print("\nEffective friction input summary:")
print(df[["True_F_fric_lub"]].describe())

print("\nParameter range check:")
print(df[PARAM_COLS].describe())

print("\nDrive-level base range check:")

base_range_df = (
    df.drop_duplicates(subset=["Drive_ID"])[BASE_COLS]
)

print(base_range_df.describe())
print("\nLubrication target range check:")
print(df[LUB_COLS].describe())

print("\nPhysics target range check:")
print(df[PHYSICS_COLS].describe())

print("\nAuxiliary column range check:")
print(df[AUX_COLS].describe())

# ---------------------------------------------------------
# Residual-based global lubrication prior
# ---------------------------------------------------------
prior_df = df[
    df["Jacking Distance"] > START_DIST
].reset_index(drop=True)

if prior_df.empty:
    raise ValueError(
        "No post-start rows are available for estimating the lubrication prior."
    )

lub_prior_info = estimate_lubrication_prior_from_residuals(prior_df)

MU_LOG_FLUB_PRIOR = lub_prior_info["mu_log_flub"]
SIGMA_LOG_FLUB_PRIOR = lub_prior_info["sigma_log_flub"]

print("\nResidual-based lubrication prior:")
print(f"  mean F_lub residual: {lub_prior_info['mean_flub']:.6f}")
print(f"  std F_lub residual:  {lub_prior_info['std_flub']:.6f}")
print(f"  cov F_lub residual:  {lub_prior_info['cov_flub']:.6f}")
print(f"  mu_log_F_lub:        {MU_LOG_FLUB_PRIOR:.6f}")
print(f"  sigma_log_F_lub:     {SIGMA_LOG_FLUB_PRIOR:.6f}")





# =========================================================
# 4. split by drive (matched to the simple-NN baseline)
# =========================================================

all_split_groups = np.array(sorted(df["Original_Drive_ID"].unique()))
split_rng = np.random.default_rng(42)
split_rng.shuffle(all_split_groups)

n_drives = len(all_split_groups)
n_train = int(n_drives * train_ratio)
n_val = int(n_drives * val_ratio)

train_split_groups = all_split_groups[:n_train]
val_split_groups = all_split_groups[n_train:n_train + n_val]
test_split_groups = all_split_groups[n_train + n_val:]

train_df = df[df["Original_Drive_ID"].isin(train_split_groups)].reset_index(drop=True)
val_df = df[df["Original_Drive_ID"].isin(val_split_groups)].reset_index(drop=True)
test_df = df[df["Original_Drive_ID"].isin(test_split_groups)].reset_index(drop=True)

print(f"\nTrain split groups: {len(train_split_groups)}, rows: {len(train_df)}")
print(f"Val split groups:   {len(val_split_groups)}, rows: {len(val_df)}")
print(f"Test split groups:  {len(test_split_groups)}, rows: {len(test_df)}")
print("Train drives:", train_df["Drive_ID"].nunique())
print("Val drives:  ", val_df["Drive_ID"].nunique())
print("Test drives: ", test_df["Drive_ID"].nunique())




# =========================================================
# 5. scalers
# =========================================================
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
scaler_P = StandardScaler()
scaler_L = StandardScaler()

X_train_raw = train_df[INPUT_COLS].values.astype(np.float32)
Y_train_raw = train_df[PARAM_COLS].values.astype(np.float32)
BASE_train_raw = train_df[BASE_COLS].values.astype(np.float32)
P_train_raw = train_df[PHYSICS_COLS].values.astype(np.float32)
A_train_raw = train_df[AUX_COLS].values.astype(np.float32)
L_train_raw = train_df[LUB_COLS].values.astype(np.float32)

X_val_raw = val_df[INPUT_COLS].values.astype(np.float32)
Y_val_raw = val_df[PARAM_COLS].values.astype(np.float32)
BASE_val_raw = val_df[BASE_COLS].values.astype(np.float32)
P_val_raw = val_df[PHYSICS_COLS].values.astype(np.float32)
A_val_raw = val_df[AUX_COLS].values.astype(np.float32)
L_val_raw = val_df[LUB_COLS].values.astype(np.float32)

X_test_raw = test_df[INPUT_COLS].values.astype(np.float32)
Y_test_raw = test_df[PARAM_COLS].values.astype(np.float32)
BASE_test_raw = test_df[BASE_COLS].values.astype(np.float32)
P_test_raw = test_df[PHYSICS_COLS].values.astype(np.float32)
A_test_raw = test_df[AUX_COLS].values.astype(np.float32)
L_test_raw = test_df[LUB_COLS].values.astype(np.float32)

LOG_L_train_raw = np.log(np.clip(L_train_raw, LUB_EPS, None))
LOG_L_val_raw = np.log(np.clip(L_val_raw, LUB_EPS, None))
LOG_L_test_raw = np.log(np.clip(L_test_raw, LUB_EPS, None))

X_train_scaled = scaler_X.fit_transform(X_train_raw)
Y_train_scaled = scaler_Y.fit_transform(Y_train_raw)
P_train_scaled = scaler_P.fit_transform(P_train_raw)
LOG_L_train_scaled = scaler_L.fit_transform(LOG_L_train_raw)

X_val_scaled = scaler_X.transform(X_val_raw)
Y_val_scaled = scaler_Y.transform(Y_val_raw)
P_val_scaled = scaler_P.transform(P_val_raw)
LOG_L_val_scaled = scaler_L.transform(LOG_L_val_raw)

X_test_scaled = scaler_X.transform(X_test_raw)
Y_test_scaled = scaler_Y.transform(Y_test_raw)
P_test_scaled = scaler_P.transform(P_test_raw)
LOG_L_test_scaled = scaler_L.transform(LOG_L_test_raw)

global_jacking_std = torch.tensor(
    train_df["Jacking Forces"].std() + 1e-6,
    dtype=torch.float32,
    device=device,
)

train_friction_obs = train_df["True_F_fric_lub"].values

global_friction_obs_std = torch.tensor(
    train_friction_obs.std() + 1e-6,
    dtype=torch.float32,
    device=device,
)

global_lub_std = torch.tensor(
    train_df["Actual_F_lub"].values.std() + 1e-6,
    dtype=torch.float32,
    device=device,
)

mu_log_flub_prior_tensor = torch.tensor(
    MU_LOG_FLUB_PRIOR,
    dtype=torch.float32,
    device=device,
)

sigma_log_flub_prior_tensor = torch.tensor(
    SIGMA_LOG_FLUB_PRIOR + 1e-8,
    dtype=torch.float32,
    device=device,
)

print("\nScaler_X input means:")
for name, val in zip(INPUT_COLS, scaler_X.mean_):
    print(f"{name:>24s}: {val:.6f}")

print("\nScaler_X input stds:")
for name, val in zip(INPUT_COLS, scaler_X.scale_):
    print(f"{name:>24s}: {val:.6f}")

print("\nScaler_Y parameter means:")
for name, val in zip(PARAM_COLS, scaler_Y.mean_):
    print(f"{name:>8s}: {val:.6f}")

print("\nScaler_Y parameter stds:")
for name, val in zip(PARAM_COLS, scaler_Y.scale_):
    print(f"{name:>8s}: {val:.6f}")

print("\nScaler_L log-lubrication mean/std:")
print(f"mean log_F_lub: {scaler_L.mean_[0]:.6f}")
print(f"std log_F_lub:  {scaler_L.scale_[0]:.6f}")

print("\nGlobal force scales:")
print(f"global_jacking_std:      {global_jacking_std.item():.6f}")
print(f"global_friction_obs_std: {global_friction_obs_std.item():.6f}")
print(f"global_lub_std:          {global_lub_std.item():.6f}")



# =========================================================
# 6. drive-aware sliding-window dataset
# =========================================================
class TunnelingCNNDataset(Dataset):

    def __init__(
        self,
        df_subset,
        X_scaled,
        Y_scaled,
        P_scaled,
        LOG_L_scaled,
        X_raw,
        Y_raw,
        BASE_raw,
        P_raw,
        L_raw,
        A_raw,
        window_size,
        min_target_position=0,
    ):
        self.df_subset = df_subset.reset_index(drop=True)

        self.X_scaled = X_scaled
        self.Y_scaled = Y_scaled
        self.P_scaled = P_scaled
        self.LOG_L_scaled = LOG_L_scaled

        self.X_raw = X_raw
        self.Y_raw = Y_raw
        self.BASE_raw = BASE_raw
        self.P_raw = P_raw
        self.L_raw = L_raw
        self.A_raw = A_raw

        self.window_size = window_size
        self.min_target_position = int(min_target_position)
        self.samples = []

        self._build_samples()

    def _build_samples(self):
        drive_ids = self.df_subset["Drive_ID"].values

        for drive_id in np.unique(drive_ids):
            indices = np.where(drive_ids == drive_id)[0]

            distance_values = self.df_subset.loc[indices, "Jacking Distance"].values
            indices = indices[np.argsort(distance_values)]

            if len(indices) < self.window_size:
                continue

            for start_pos in range(0, len(indices) - self.window_size + 1):
                target_position = start_pos + self.window_size - 1

                # Use common downstream target positions for all tested
                # window sizes. This prevents larger windows from being
                # evaluated on a different target subset.
                if target_position < self.min_target_position:
                    continue

                window_indices = indices[start_pos:start_pos + self.window_size]
                target_idx = window_indices[-1]
                self.samples.append((window_indices, target_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window_indices, target_idx = self.samples[idx]

        x_seq_scaled = self.X_scaled[window_indices]

        y_target_scaled = self.Y_scaled[target_idx]
        p_target_scaled = self.P_scaled[target_idx]
        log_lub_target_scaled = self.LOG_L_scaled[target_idx]

        x_target_raw = self.X_raw[target_idx]
        y_target_raw = self.Y_raw[target_idx]
        base_target_raw = self.BASE_raw[target_idx]
        p_target_raw = self.P_raw[target_idx]
        lub_target_raw = self.L_raw[target_idx]
        a_target_raw = self.A_raw[target_idx]

        return (
            torch.tensor(x_seq_scaled, dtype=torch.float32),
            torch.tensor(y_target_scaled, dtype=torch.float32),
            torch.tensor(p_target_scaled, dtype=torch.float32),
            torch.tensor(log_lub_target_scaled, dtype=torch.float32),
            torch.tensor(x_target_raw, dtype=torch.float32),
            torch.tensor(y_target_raw, dtype=torch.float32),
            torch.tensor(base_target_raw, dtype=torch.float32),
            torch.tensor(p_target_raw, dtype=torch.float32),
            torch.tensor(lub_target_raw, dtype=torch.float32),
            torch.tensor(a_target_raw, dtype=torch.float32),
        )


train_dataset = TunnelingCNNDataset(
    train_df,
    X_train_scaled,
    Y_train_scaled,
    P_train_scaled,
    LOG_L_train_scaled,
    X_train_raw,
    Y_train_raw,
    BASE_train_raw,
    P_train_raw,
    L_train_raw,
    A_train_raw,
    window_size,
    min_target_position=COMMON_TARGET_START_POS,
)

val_dataset = TunnelingCNNDataset(
    val_df,
    X_val_scaled,
    Y_val_scaled,
    P_val_scaled,
    LOG_L_val_scaled,
    X_val_raw,
    Y_val_raw,
    BASE_val_raw,
    P_val_raw,
    L_val_raw,
    A_val_raw,
    window_size,
    min_target_position=COMMON_TARGET_START_POS,
)

test_dataset = TunnelingCNNDataset(
    test_df,
    X_test_scaled,
    Y_test_scaled,
    P_test_scaled,
    LOG_L_test_scaled,
    X_test_raw,
    Y_test_raw,
    BASE_test_raw,
    P_test_raw,
    L_test_raw,
    A_test_raw,
    window_size,
    min_target_position=COMMON_TARGET_START_POS,
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("\nTrain windows:", len(train_dataset))
print("Val windows:  ", len(val_dataset))
print("Test windows: ", len(test_dataset))
print("Window size:   ", window_size)
print("Batch size:    ", batch_size)
print("Common target start position:", COMMON_TARGET_START_POS)





# =========================================================
# 7. conditional-head MHNN model with early latent lubrication head
# =========================================================
class MultiHeadCNNInverseModel(nn.Module):
    """
    Shared NN backbone + conditional parameter heads + auxiliary physics head
    + early latent lubrication head.

    Main structure:
        z -> N0
        z -> log_F_lub
        [z, N0, log_F_lub] -> mu
        [z, N0, log_F_lub, mu] -> phi
        [z, N0, log_F_lub, mu, phi] -> gamma

    """

    def __init__(
        self,
        input_features,
        sequence_length,
        dropout,
        y_mean,
        y_scale,
        log_lub_mean,
        log_lub_scale,
    ):
        super().__init__()

        self.register_buffer(
            "y_mean",
            torch.tensor(y_mean, dtype=torch.float32).view(1, -1),
        )

        self.register_buffer(
            "y_scale",
            torch.tensor(y_scale, dtype=torch.float32).view(1, -1),
        )

        self.register_buffer(
            "log_lub_mean",
            torch.tensor(log_lub_mean, dtype=torch.float32).view(1, -1),
        )

        self.register_buffer(
            "log_lub_scale",
            torch.tensor(log_lub_scale, dtype=torch.float32).view(1, -1),
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(input_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        flattened_size = 128 * sequence_length

        self.shared_fc = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # First-level heads from shared representation z.
        self.N0_head = nn.Linear(64, 1)
        self.log_lub_head = nn.Linear(64, 1)

        # Friction-related conditional heads.
        self.mu_head = nn.Linear(64 + 2, 1)
        self.phi_head = nn.Linear(64 + 3, 1)
        self.gamma_head = nn.Linear(64 + 4, 1)

        self.physics_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def raw_to_scaled(self, raw_value, param_index):
        return (
            raw_value - self.y_mean[:, param_index:param_index + 1]
        ) / self.y_scale[:, param_index:param_index + 1]

    def scaled_log_lub_to_raw(self, log_lub_scaled):
        return log_lub_scaled * self.log_lub_scale + self.log_lub_mean

    def bounded_dynamic_residual(
        self,
        score,
        base_raw,
        relative_cv,
    ):
        """
        Return a bounded residual around a drive-specific base value.

        """
        eps = 1e-6
        base_safe = base_raw
        half_range = relative_cv * torch.abs(base_safe)
        lower_bound = base_safe - half_range
        upper_bound = base_safe + half_range

        unit = torch.tanh(score)
        delta_low = base_safe - lower_bound
        delta_high = upper_bound - base_safe

        delta = torch.where(
            unit >= 0.0,
            unit * delta_high,
            unit * delta_low,
        )

        return base_safe + delta

    def forward(
        self,
        x,
        detach_previous=True,
        y_target_scaled=None,
        log_lub_target_scaled=None,
        base_target_raw=None,
        training_stage=5,
        conditional_placeholder_mode="predicted",
    ):


        if conditional_placeholder_mode not in {"true", "predicted"}:
            raise ValueError(
                "conditional_placeholder_mode must be either "
                "'true' or 'predicted'."
            )

        use_true_conditional_inputs = stage_uses_true_conditional_inputs(
            training_stage,
            mode=conditional_placeholder_mode,
        )

        if use_true_conditional_inputs:
            if y_target_scaled is None or log_lub_target_scaled is None:
                raise ValueError(
                    "Teacher-forced conditional inputs require "
                    "y_target_scaled and log_lub_target_scaled."
                )

        if base_target_raw is None:
            raise ValueError(
                "Dataset-provided base_target_raw is required for the "
                "bounded residual phi/gamma heads."
            )

        x = x.permute(0, 2, 1)

        z = self.conv_block(x)
        z = z.reshape(z.size(0), -1)
        z = self.shared_fc(z)

        # -------------------------------------------------
        # 1. Predict N0 from z.
        # -------------------------------------------------
        N0_scaled = self.N0_head(z)

        if use_true_conditional_inputs:
            N0_for_next = y_target_scaled[:, 0:1].detach()
        elif detach_previous:
            N0_for_next = N0_scaled.detach()
        else:
            N0_for_next = N0_scaled

        # -------------------------------------------------
        # 2. Predict latent lubrication from z.
        # -------------------------------------------------
        log_lub_scaled = self.log_lub_head(z)
        log_lub_raw = self.scaled_log_lub_to_raw(log_lub_scaled)
        F_lub_raw = torch.exp(log_lub_raw)

        # -------------------------------------------------
        # 2b. Use the drive-level dataset bases directly.
        # -------------------------------------------------
        base_phi_used_raw = base_target_raw[:, 0:1].detach()
        base_gamma_used_raw = base_target_raw[:, 1:2].detach()

        if use_true_conditional_inputs:
            log_lub_for_next = log_lub_target_scaled.detach()
        elif detach_previous:
            log_lub_for_next = log_lub_scaled.detach()
        else:
            log_lub_for_next = log_lub_scaled

        # -------------------------------------------------
        # 3. Predict mu from [z, N0, log_F_lub].
        # -------------------------------------------------
        mu_input = torch.cat(
            [z, N0_for_next, log_lub_for_next],
            dim=1,
        )

        mu_scaled = self.mu_head(mu_input)

        if use_true_conditional_inputs:
            mu_for_next = y_target_scaled[:, 1:2].detach()
        elif detach_previous:
            mu_for_next = mu_scaled.detach()
        else:
            mu_for_next = mu_scaled

        # -------------------------------------------------
        # 4. Predict phi from [z, N0, log_F_lub, mu].
        # -------------------------------------------------
        phi_input = torch.cat(
            [z, N0_for_next, log_lub_for_next, mu_for_next],
            dim=1,
        )

        phi_score = self.phi_head(phi_input)

        phi_raw = self.bounded_dynamic_residual(
            score=phi_score,
            base_raw=base_phi_used_raw,
            relative_cv=DRIVE_BOUND_CV_PHI,
        )

        phi_scaled = self.raw_to_scaled(
            raw_value=phi_raw,
            param_index=2,
        )

        if use_true_conditional_inputs and training_stage == 5:
            phi_for_next = y_target_scaled[:, 2:3].detach()
        elif detach_previous:
            phi_for_next = phi_scaled.detach()
        else:
            phi_for_next = phi_scaled

        # -------------------------------------------------
        # 5. Predict gamma from [z, N0, log_F_lub, mu, phi].
        # -------------------------------------------------
        gamma_input = torch.cat(
            [z, N0_for_next, log_lub_for_next, mu_for_next, phi_for_next],
            dim=1,
        )

        gamma_score = self.gamma_head(gamma_input)

        gamma_raw = self.bounded_dynamic_residual(
            score=gamma_score,
            base_raw=base_gamma_used_raw,
            relative_cv=DRIVE_BOUND_CV_GAMMA,
        )

        gamma_scaled = self.raw_to_scaled(
            raw_value=gamma_raw,
            param_index=3,
        )

        params_scaled = torch.cat(
            [N0_scaled, mu_scaled, phi_scaled, gamma_scaled],
            dim=1,
        )

        physics_scaled = self.physics_head(z)

        return (
            params_scaled,
            physics_scaled,
            log_lub_scaled,
            log_lub_raw,
            F_lub_raw,
        )


model = MultiHeadCNNInverseModel(
    input_features=len(INPUT_COLS),
    sequence_length=window_size,
    dropout=dropout,
    y_mean=scaler_Y.mean_,
    y_scale=scaler_Y.scale_,
    log_lub_mean=scaler_L.mean_,
    log_lub_scale=scaler_L.scale_,
).to(device)

print("\nModel:")
print(model)




# =========================================================
# 8. stage configuration
# =========================================================
def get_stage_loss_config(training_stage, device):
    """
    Parameter order:
        [N0, mu, phi_deg, gamma]

    Staged order:
        Stage 1: N0
        Stage 2: latent lubrication
        Stage 3: mu
        Stage 4: phi
        Stage 5: gamma
    """

    if training_stage == 1:
        param_weights = [1.0, 0.0, 0.0, 0.0]
        phys_weights = [0.0, 0.0, 0.0, 0.0]

        cons_face_weight = 0.0
        cons_kfric_weight = 0.0

        total_force_weight = 0.0
        friction_obs_weight = 0.0

        face_target_weight = 1.0
        kfric_target_weight = 0.0

        phi_prior_weight = 0.0
        gamma_prior_weight = 0.0

        lub_supervised_weight = 0.0
        lub_prior_weight = 0.0
        lub_raw_force_weight = 0.0

    elif training_stage == 2:
        param_weights = [0.2, 0.0, 0.0, 0.0]
        phys_weights = [0.0, 0.0, 0.0, 0.0]

        cons_face_weight = 0.0
        cons_kfric_weight = 0.0

        total_force_weight = 1.0
        friction_obs_weight = 1.0

        face_target_weight = 0.5
        kfric_target_weight = 0.0

        phi_prior_weight = 0.0
        gamma_prior_weight = 0.0

        lub_supervised_weight = 1.0
        lub_prior_weight = 0.0
        lub_raw_force_weight = 1.0

    elif training_stage == 3:
        param_weights = [0.2, 1.0, 0.0, 0.0]
        phys_weights = [1.0, 1.0, 1.0, 1.0]

        cons_face_weight = 0.0
        cons_kfric_weight = 1.0

        total_force_weight = 1.0
        friction_obs_weight = 1.0

        face_target_weight = 0.5
        kfric_target_weight = 1.0

        phi_prior_weight = 0.0
        gamma_prior_weight = 0.0

        lub_supervised_weight = 0.5
        lub_prior_weight = 0.0
        lub_raw_force_weight = 0.5

    elif training_stage == 4:
        param_weights = [0.2, 0.2, 1.0, 0.0]
        phys_weights = [0.0, 0.0, 0.0, 0.0]

        cons_face_weight = 0.0
        cons_kfric_weight = 0.0

        total_force_weight = 1.0
        friction_obs_weight = 1.0

        face_target_weight = 0.5
        kfric_target_weight = 0.0

        phi_prior_weight = 1.0
        gamma_prior_weight = 0.0

        lub_supervised_weight = 0.5
        lub_prior_weight = 0.0
        lub_raw_force_weight = 0.5

    elif training_stage == 5:
        param_weights = [0.5, 0.5, 0.5, 1.0]
        phys_weights = [0.5, 0.2, 0.5, 0.5]

        cons_face_weight = 0.5
        cons_kfric_weight = 0.2

        total_force_weight = 1.0
        friction_obs_weight = 1.0

        face_target_weight = 0.5
        kfric_target_weight = 0.0

        phi_prior_weight = 0.5
        gamma_prior_weight = 1.0

        lub_supervised_weight = 0.5
        lub_prior_weight = 0.0
        lub_raw_force_weight = 0.5

    else:
        raise ValueError("training_stage must be an integer from 1 through 5.")

    return {
        "param_weights": torch.tensor(param_weights, dtype=torch.float32, device=device),
        "phys_weights": torch.tensor(phys_weights, dtype=torch.float32, device=device),
        "cons_face_weight": cons_face_weight,
        "cons_kfric_weight": cons_kfric_weight,
        "total_force_weight": total_force_weight,
        "friction_obs_weight": friction_obs_weight,
        "face_target_weight": face_target_weight,
        "kfric_target_weight": kfric_target_weight,
        "phi_prior_weight": phi_prior_weight,
        "gamma_prior_weight": gamma_prior_weight,
        "lub_supervised_weight": lub_supervised_weight,
        "lub_prior_weight": lub_prior_weight,
        "lub_raw_force_weight": lub_raw_force_weight,
    }


def get_stage_learning_rates(training_stage):
    if training_stage == 1:
        return {
            "backbone": 1e-3,
            "N0": 1e-3,
            "mu": 0.0,
            "phi": 0.0,
            "gamma": 0.0,
            "lub": 0.0,
            "physics": 0.0,
        }

    elif training_stage == 2:

        return {
            "backbone": 1e-4,
            "N0": 0.0,
            "mu": 0.0,
            "phi": 0.0,
            "gamma": 0.0,
            "lub": 1e-3,
            "physics": 0.0,
        }

    elif training_stage == 3:

        return {
            "backbone": 0.0,
            "N0": 0.0,
            "mu": 1e-3,
            "phi": 0.0,
            "gamma": 0.0,
            "lub": 0.0,
            "physics": 1e-5,
        }

    elif training_stage == 4:
        return {
            "backbone": 0.0,
            "N0": 0.0,
            "mu": 0.0,
            "phi": 1e-3,
            "gamma": 0.0,
            "lub": 1e-5,
            "physics": 0.0,
        }

    elif training_stage == 5:
        return {
            "backbone": 0.0,
            "N0": 0.0,
            "mu": 0.0,
            "phi": 0.0,
            "gamma": 1e-3,
            "lub": 1e-5,
            "physics": 1e-4,
        }

    else:
        raise ValueError("training_stage must be an integer from 1 through 5.")




# =========================================================
# 9. helper functions
# =========================================================
def inverse_transform_torch(y_scaled, scaler, device):
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)
    return y_scaled * scale + mean


def weighted_loss_from_col_losses(col_losses, weights):
    denom = torch.sum(weights)

    if denom.item() <= 0:
        return torch.tensor(0.0, dtype=torch.float32, device=col_losses.device)

    return torch.sum(col_losses * weights) / (denom + 1e-8)


def weighted_mse_by_column(pred, target, weights):
    col_losses = torch.mean((pred - target) ** 2, dim=0)
    loss = weighted_loss_from_col_losses(col_losses, weights)
    return loss, col_losses


def reconstruct_face_and_kfric_from_params(params_raw):
    """
    Reconstruct:
        1. F_face from N0
        2. K_fric_local from mu, phi, gamma
    """

    N0 = params_raw[:, 0:1]
    mu = params_raw[:, 1:2]
    phi_deg = params_raw[:, 2:3]
    gamma = params_raw[:, 3:4]

    phi_rad = phi_deg * torch.pi / 180.0

    f_face = 10.0 * 1.32 * torch.pi * tbm_diameter * N0
    F_face_formula = f_face * torch.pi * (tbm_diameter / 2.0) ** 2

    f_fric_local = (
        mu
        * gamma
        * pipe_diameter
        * torch.cos(torch.pi / 4.0 + phi_rad / 2.0)
        / (2.0 * torch.tan(phi_rad))
    )

    K_fric_formula = f_fric_local * torch.pi * pipe_diameter

    return F_face_formula, K_fric_formula


def build_stage_physics_params(
    params_pred_raw,
    y_target_raw,
    training_stage,
    placeholder_mode="true",
    use_true_conditional_inputs=False,
):


    N0_pred = params_pred_raw[:, 0:1]
    mu_pred = params_pred_raw[:, 1:2]
    phi_pred = params_pred_raw[:, 2:3]
    gamma_pred = params_pred_raw[:, 3:4]

    if placeholder_mode == "true":
        N0_ref = y_target_raw[:, 0:1]
        mu_ref = y_target_raw[:, 1:2]
        phi_ref = y_target_raw[:, 2:3]
        gamma_ref = y_target_raw[:, 3:4]

    elif placeholder_mode == "fixed":
        batch_size_now = params_pred_raw.shape[0]

        N0_ref = torch.full(
            (batch_size_now, 1),
            REF_N0,
            dtype=torch.float32,
            device=params_pred_raw.device,
        )

        mu_ref = torch.full(
            (batch_size_now, 1),
            REF_MU,
            dtype=torch.float32,
            device=params_pred_raw.device,
        )

        phi_ref = torch.full(
            (batch_size_now, 1),
            REF_PHI_DEG,
            dtype=torch.float32,
            device=params_pred_raw.device,
        )

        gamma_ref = torch.full(
            (batch_size_now, 1),
            REF_GAMMA,
            dtype=torch.float32,
            device=params_pred_raw.device,
        )

    else:
        raise ValueError("placeholder_mode must be either 'true' or 'fixed'")

    if training_stage <= 2:
        N0_used = N0_pred
        mu_used = mu_ref
        phi_used = phi_ref
        gamma_used = gamma_ref

    elif training_stage == 3:
        N0_used = y_target_raw[:, 0:1] if use_true_conditional_inputs else N0_pred
        mu_used = mu_pred
        phi_used = phi_ref
        gamma_used = gamma_ref

    elif training_stage == 4:
        N0_used = y_target_raw[:, 0:1] if use_true_conditional_inputs else N0_pred
        mu_used = y_target_raw[:, 1:2] if use_true_conditional_inputs else mu_pred
        phi_used = phi_pred
        gamma_used = gamma_ref

    elif training_stage == 5:
        N0_used = y_target_raw[:, 0:1] if use_true_conditional_inputs else N0_pred
        mu_used = y_target_raw[:, 1:2] if use_true_conditional_inputs else mu_pred
        phi_used = y_target_raw[:, 2:3] if use_true_conditional_inputs else phi_pred
        gamma_used = gamma_pred

    else:
        raise ValueError("training_stage must be an integer from 1 through 5.")

    params_used = torch.cat(
        [N0_used, mu_used, phi_used, gamma_used],
        dim=1,
    )

    return params_used


def compute_fixed_start_friction(batch_size_now, device):
    phi_rad = torch.tensor(
        REF_PHI_DEG * np.pi / 180.0,
        dtype=torch.float32,
        device=device,
    )

    f_fric_ref = (
        REF_MU
        * REF_GAMMA
        * pipe_diameter
        * torch.cos(torch.pi / 4.0 + phi_rad / 2.0)
        / (2.0 * torch.tan(phi_rad))
    )

    K_fric_ref = f_fric_ref * torch.pi * pipe_diameter
    F_fric_at_start_ref = K_fric_ref * START_DIST

    return torch.full(
        (batch_size_now, 1),
        F_fric_at_start_ref.item(),
        dtype=torch.float32,
        device=device,
    )


def reconstruct_force_from_stage_params(
    params_used_for_physics,
    x_target_raw,
    a_target_raw,
    F_lub_pred_raw,
    placeholder_mode="true",
):


    F_face_formula, K_fric_formula = reconstruct_face_and_kfric_from_params(
        params_used_for_physics
    )

    jacking_obs = x_target_raw[
        :,
        INPUT_IDX["Jacking Forces"]:INPUT_IDX["Jacking Forces"] + 1,
    ]

    F_face_input = x_target_raw[
        :,
        INPUT_IDX["F_face"]:INPUT_IDX["F_face"] + 1,
    ]

    if "True_F_fric_lub" in INPUT_IDX:
        friction_obs = x_target_raw[
            :,
            INPUT_IDX["True_F_fric_lub"]:INPUT_IDX["True_F_fric_lub"] + 1,
        ]
    else:
        raise KeyError(
            "True_F_fric_lub is not found in INPUT_IDX. "
            "For Version 4.1.10, add 'True_F_fric_lub' to INPUT_COLS."
        )

    jacking_distance = x_target_raw[
        :,
        INPUT_IDX["Jacking Distance"]:INPUT_IDX["Jacking Distance"] + 1,
    ]
    local_distance_from_start = torch.clamp(
        jacking_distance - START_DIST,
        min=0.0,
    )

    if placeholder_mode == "true":
        F_fric_at_start = a_target_raw[:, 1:2]

    elif placeholder_mode == "fixed":
        F_fric_at_start = compute_fixed_start_friction(
            batch_size_now=x_target_raw.shape[0],
            device=x_target_raw.device,
        )

    else:
        raise ValueError("placeholder_mode must be either 'true' or 'fixed'")

    F_fric_native_after_start = (
        F_fric_at_start
        + K_fric_formula * local_distance_from_start
    )

    F_fric_native_before_start = (
        K_fric_formula * torch.clamp(jacking_distance, min=0.0)
    )

    F_fric_native_formula = torch.where(
        jacking_distance <= START_DIST,
        F_fric_native_before_start,
        F_fric_native_after_start,
    )

    F_fric_effective_formula = (
        F_fric_native_formula
        - F_lub_pred_raw
    )

    F_jacking_formula = (
        F_face_input
        + F_fric_effective_formula
    )

    return {
        "F_jacking_formula": F_jacking_formula,
        "F_face_formula": F_face_formula,
        "F_face_input": F_face_input,
        "K_fric_formula": K_fric_formula,
        "F_fric_native_formula": F_fric_native_formula,
        "F_fric_effective_formula": F_fric_effective_formula,
        "jacking_obs": jacking_obs,
        "friction_obs": friction_obs,
    }


def soft_bound_loss(params_raw, param_weights):
    N0 = params_raw[:, 0]
    mu = params_raw[:, 1]
    phi = params_raw[:, 2]
    gamma = params_raw[:, 3]

    N0_low, N0_high = BOUNDS["N0"]
    mu_low, mu_high = BOUNDS["mu"]
    phi_low, phi_high = BOUNDS["phi_deg"]
    gamma_low, gamma_high = BOUNDS["gamma"]

    bound_losses = torch.stack([
        torch.relu(N0_low - N0).pow(2).mean()
        + torch.relu(N0 - N0_high).pow(2).mean(),

        torch.relu(mu_low - mu).pow(2).mean()
        + torch.relu(mu - mu_high).pow(2).mean(),

        torch.relu(phi_low - phi).pow(2).mean()
        + torch.relu(phi - phi_high).pow(2).mean(),

        torch.relu(gamma_low - gamma).pow(2).mean()
        + torch.relu(gamma - gamma_high).pow(2).mean(),
    ])

    return weighted_loss_from_col_losses(bound_losses, param_weights)


def dynamic_residual_prior_loss(
    params_pred_raw,
    base_phi_used_raw,
    base_gamma_used_raw,
    config,
):

    phi_pred = params_pred_raw[:, 2:3]
    gamma_pred = params_pred_raw[:, 3:4]

    phi_base = base_phi_used_raw.detach()
    gamma_base = base_gamma_used_raw.detach()

    phi_scale = DRIVE_BOUND_CV_PHI * torch.abs(phi_base)
    gamma_scale = DRIVE_BOUND_CV_GAMMA * torch.abs(gamma_base)

    phi_error = (phi_pred - phi_base) / (phi_scale + 1e-8)
    gamma_error = (gamma_pred - gamma_base) / (gamma_scale + 1e-8)

    loss_phi_prior = torch.mean(phi_error ** 2)
    loss_gamma_prior = torch.mean(gamma_error ** 2)

    loss_residual_prior = (
        config["phi_prior_weight"] * loss_phi_prior
        + config["gamma_prior_weight"] * loss_gamma_prior
    )

    return loss_residual_prior, loss_phi_prior, loss_gamma_prior


def lubrication_prior_loss(log_lub_pred_raw, config):


    standardized_error = (
        log_lub_pred_raw - mu_log_flub_prior_tensor
    ) / sigma_log_flub_prior_tensor

    loss = torch.mean(standardized_error ** 2)
    return config["lub_prior_weight"] * loss, loss


def engineering_tolerance_metrics(params_pred_raw, params_true_raw):
    phi_pred = params_pred_raw[:, 2]
    phi_true = params_true_raw[:, 2]

    gamma_pred = params_pred_raw[:, 3]
    gamma_true = params_true_raw[:, 3]

    phi_abs_error = torch.abs(phi_pred - phi_true)
    gamma_abs_error = torch.abs(gamma_pred - gamma_true)

    metrics = {}

    metrics["phi_mae_deg"] = phi_abs_error.mean()
    metrics["gamma_mae"] = gamma_abs_error.mean()

    metrics["phi_within_tol"] = (phi_abs_error <= PHI_TOL_DEG).float().mean()
    metrics["gamma_within_tol"] = (gamma_abs_error <= GAMMA_TOL).float().mean()

    for tol in PHI_TOL_LIST:
        metrics[tol_key("phi", tol)] = (phi_abs_error <= tol).float().mean()

    for tol in GAMMA_TOL_LIST:
        metrics[tol_key("gamma", tol)] = (gamma_abs_error <= tol).float().mean()

    return metrics


def gated_loss(loss_value, enabled):

    if enabled:
        return loss_value

    return loss_value * 0.0



# =========================================================
# 10. loss function
# =========================================================
def compute_loss(
    model,
    x_seq_scaled,
    y_target_scaled,
    base_target_raw,
    p_target_scaled,
    log_lub_target_scaled,
    x_target_raw,
    y_target_raw,
    p_target_raw,
    lub_target_raw,
    a_target_raw,
    training_stage,
):
    detach_previous = training_stage <= 5

    (
        params_pred_scaled,
        physics_pred_scaled,
        log_lub_pred_scaled,
        log_lub_pred_raw,
        F_lub_pred_raw,
    ) = model(
        x_seq_scaled,
        detach_previous=detach_previous,
        y_target_scaled=y_target_scaled,
        log_lub_target_scaled=log_lub_target_scaled,
        base_target_raw=base_target_raw,
        training_stage=training_stage,
        conditional_placeholder_mode=CONDITIONAL_PLACEHOLDER_MODE,
    )

    params_pred_raw = inverse_transform_torch(
        params_pred_scaled,
        scaler_Y,
        device,
    )

    physics_pred_raw = inverse_transform_torch(
        physics_pred_scaled,
        scaler_P,
        device,
    )

    use_true_stage_inputs = stage_uses_true_conditional_inputs(
        training_stage
    )

    F_lub_for_physics = (
        lub_target_raw
        if use_true_stage_inputs
        else F_lub_pred_raw
    )

    config = get_stage_loss_config(
        training_stage,
        device,
    )

    param_weights = config["param_weights"]
    phys_weights = config["phys_weights"]

    # -----------------------------------------------------
    # A. parameter loss
    # -----------------------------------------------------
    loss_param, param_col_losses = weighted_mse_by_column(
        params_pred_scaled,
        y_target_scaled,
        param_weights,
    )

    tol_metrics = engineering_tolerance_metrics(
        params_pred_raw=params_pred_raw,
        params_true_raw=y_target_raw,
    )

    base_phi_used_raw = base_target_raw[:, 0:1].detach()
    base_gamma_used_raw = base_target_raw[:, 1:2].detach()

    # -----------------------------------------------------
    # A1. dynamic phi/gamma residual prior
    # -----------------------------------------------------
    (
        loss_residual_prior,
        loss_phi_prior,
        loss_gamma_prior,
    ) = dynamic_residual_prior_loss(
        params_pred_raw=params_pred_raw,
        base_phi_used_raw=base_phi_used_raw,
        base_gamma_used_raw=base_gamma_used_raw,
        config=config,
    )

    # -----------------------------------------------------
    # A2. direct lubrication supervision
    # -----------------------------------------------------
    loss_lub_supervised_raw = torch.mean(
        (
            log_lub_pred_scaled
            - log_lub_target_scaled
        ) ** 2
    )

    loss_lub_supervised = (
        config["lub_supervised_weight"]
        * loss_lub_supervised_raw
    )

    # -----------------------------------------------------
    # A3. lubrication prior
    # -----------------------------------------------------
    loss_lub_prior, loss_lub_prior_raw = (
        lubrication_prior_loss(
            log_lub_pred_raw=log_lub_pred_raw,
            config=config,
        )
    )

    # -----------------------------------------------------
    # A4. raw lubrication-force supervision
    # -----------------------------------------------------
    loss_lub_raw_force_raw = torch.mean(
        (
            (F_lub_pred_raw - lub_target_raw)
            / global_lub_std
        ) ** 2
    )

    loss_lub_raw_force = (
        config["lub_raw_force_weight"]
        * loss_lub_raw_force_raw
    )

    # -----------------------------------------------------
    # B. auxiliary physics-head loss
    # -----------------------------------------------------
    loss_phys_direct, phys_col_losses = (
        weighted_mse_by_column(
            physics_pred_scaled,
            p_target_scaled,
            phys_weights,
        )
    )

    # -----------------------------------------------------
    # C. staged physical reconstruction
    # -----------------------------------------------------
    params_used_for_physics = (
        build_stage_physics_params(
            params_pred_raw=params_pred_raw,
            y_target_raw=y_target_raw,
            training_stage=training_stage,
            placeholder_mode=PLACEHOLDER_MODE,
            use_true_conditional_inputs=(
                use_true_stage_inputs
            ),
        )
    )

    force_terms = reconstruct_force_from_stage_params(
        params_used_for_physics=params_used_for_physics,
        x_target_raw=x_target_raw,
        a_target_raw=a_target_raw,
        F_lub_pred_raw=F_lub_for_physics,
        placeholder_mode=PLACEHOLDER_MODE,
    )

    F_jacking_formula = force_terms["F_jacking_formula"]
    F_face_formula = force_terms["F_face_formula"]
    K_fric_formula = force_terms["K_fric_formula"]
    F_fric_effective_formula = (
        force_terms["F_fric_effective_formula"]
    )

    jacking_obs = force_terms["jacking_obs"]
    friction_obs = force_terms["friction_obs"]

    # -----------------------------------------------------
    # D. analytical-to-physics-head consistency
    # -----------------------------------------------------
    F_face_direct = physics_pred_raw[:, 0:1]
    K_fric_direct = physics_pred_raw[:, 1:2]

    F_face_scale = torch.tensor(
        scaler_P.scale_[0],
        dtype=torch.float32,
        device=device,
    )

    K_fric_scale = torch.tensor(
        scaler_P.scale_[1],
        dtype=torch.float32,
        device=device,
    )

    loss_cons_face_raw = torch.mean(
        (
            (F_face_direct - F_face_formula)
            / F_face_scale
        ) ** 2
    )

    loss_cons_kfric_raw = torch.mean(
        (
            (K_fric_direct - K_fric_formula)
            / K_fric_scale
        ) ** 2
    )

    loss_cons_face = (
        config["cons_face_weight"]
        * loss_cons_face_raw
    )

    loss_cons_kfric = (
        config["cons_kfric_weight"]
        * loss_cons_kfric_raw
    )

    loss_consistency = (
        loss_cons_face
        + loss_cons_kfric
    )

    # -----------------------------------------------------
    # E. basic total-jacking-force loss
    # -----------------------------------------------------
    loss_total_force_raw = torch.mean(
        (
            (F_jacking_formula - jacking_obs)
            / global_jacking_std
        ) ** 2
    )

    loss_total_force = (
        config["total_force_weight"]
        * loss_total_force_raw
    )

    # -----------------------------------------------------
    # E2. basic effective-friction loss
    # -----------------------------------------------------
    loss_friction_obs_raw = torch.mean(
        (
            (F_fric_effective_formula - friction_obs)
            / global_friction_obs_std
        ) ** 2
    )

    loss_friction_obs = (
        config["friction_obs_weight"]
        * loss_friction_obs_raw
    )

    # -----------------------------------------------------
    # F. basic face-resistance loss for N0
    # -----------------------------------------------------
    F_face_target = p_target_raw[:, 0:1]

    loss_face_target_raw = torch.mean(
        (
            (F_face_formula - F_face_target)
            / F_face_scale
        ) ** 2
    )

    loss_face_target = (
        config["face_target_weight"]
        * loss_face_target_raw
    )

    # -----------------------------------------------------
    # F2. additional K_fric target loss
    # -----------------------------------------------------
    K_fric_target = p_target_raw[:, 1:2]

    loss_kfric_target_raw = torch.mean(
        (
            (K_fric_formula - K_fric_target)
            / K_fric_scale
        ) ** 2
    )

    loss_kfric_target = (
        config["kfric_target_weight"]
        * lambda_face_target
        * loss_kfric_target_raw
    )

    # -----------------------------------------------------
    # G. physical-bound loss
    # -----------------------------------------------------
    loss_bound = soft_bound_loss(
        params_pred_raw,
        param_weights,
    )

    # -----------------------------------------------------
    # H. separate gradient contributions
    # -----------------------------------------------------

    # H1. Direct parameter supervision
    parameter_supervision_contribution = gated_loss(
        lambda_param * loss_param,
        USE_PARAMETER_SUPERVISION_IN_GRADIENT,
    )

    # H2. Direct lubrication supervision
    # This is report-only when the switch is False.
    lubrication_supervision_contribution = gated_loss(
        lambda_lub_supervised * loss_lub_supervised
        + lambda_lub_raw_force * loss_lub_raw_force,
        USE_LUBRICATION_SUPERVISION_IN_GRADIENT,
    )

    # H3. Basic face-resistance loss
    basic_face_force_contribution = gated_loss(
        lambda_face_target * loss_face_target,
        USE_BASIC_FACE_FORCE_IN_GRADIENT,
    )

    # H4. Basic friction-force loss
    basic_friction_force_contribution = gated_loss(
        lambda_friction_obs * loss_friction_obs,
        USE_BASIC_FRICTION_FORCE_IN_GRADIENT,
    )

    # H5. Basic total-jacking-force loss
    basic_total_force_contribution = gated_loss(
        lambda_total_force * loss_total_force,
        USE_BASIC_TOTAL_FORCE_IN_GRADIENT,
    )

    # H6. Additional physics-informed losses.
    # The three basic force losses are intentionally excluded here.
    physics_informed_contribution = gated_loss(
        lambda_phys_direct * loss_phys_direct
        + lambda_consistency * loss_consistency
        + loss_kfric_target
        + lambda_bound * loss_bound
        + lambda_residual_prior * loss_residual_prior
        + lambda_lub_prior * loss_lub_prior,
        USE_PHYSICS_INFORMED_LOSS_IN_GRADIENT,
    )

    # H7. Final backpropagated loss
    total_loss = (
        parameter_supervision_contribution
        + lubrication_supervision_contribution
        + basic_face_force_contribution
        + basic_friction_force_contribution
        + basic_total_force_contribution
        + physics_informed_contribution
    )

    # -----------------------------------------------------
    # I. reporting metrics
    # -----------------------------------------------------
    lub_abs_error = torch.abs(
        F_lub_pred_raw - lub_target_raw
    )

    lub_mae = torch.mean(lub_abs_error)

    loss_dict = {
        "total": total_loss,
        "param": loss_param,

        "parameter_supervision_contribution": (
            parameter_supervision_contribution
        ),

        "lubrication_supervision_contribution": (
            lubrication_supervision_contribution
        ),

        "basic_face_force_contribution": (
            basic_face_force_contribution
        ),

        "basic_friction_force_contribution": (
            basic_friction_force_contribution
        ),

        "basic_total_force_contribution": (
            basic_total_force_contribution
        ),

        "physics_informed_contribution": (
            physics_informed_contribution
        ),

        "phys_direct": loss_phys_direct,
        "cons_face": loss_cons_face,
        "cons_kfric": loss_cons_kfric,
        "consistency": loss_consistency,

        "total_force": loss_total_force,
        "friction_obs": loss_friction_obs,
        "face_target": loss_face_target,
        "kfric_target": loss_kfric_target,
        "bound": loss_bound,

        "residual_prior": (
            lambda_residual_prior
            * loss_residual_prior
        ),

        "residual_prior_raw": loss_residual_prior,
        "phi_prior": loss_phi_prior,
        "gamma_prior": loss_gamma_prior,

        "lub_supervised": loss_lub_supervised,
        "lub_prior": loss_lub_prior,
        "lub_supervised_raw": loss_lub_supervised_raw,
        "lub_prior_raw": loss_lub_prior_raw,
        "lub_mae": lub_mae,
        "lub_raw_force": loss_lub_raw_force,
        "lub_raw_force_raw": loss_lub_raw_force_raw,

        "param_N0": param_col_losses[0],
        "param_mu": param_col_losses[1],
        "param_phi": param_col_losses[2],
        "param_gamma": param_col_losses[3],

        "phi_mae_deg": tol_metrics["phi_mae_deg"],
        "gamma_mae": tol_metrics["gamma_mae"],
        "phi_within_tol": tol_metrics["phi_within_tol"],
        "gamma_within_tol": tol_metrics["gamma_within_tol"],

        "phys_face": phys_col_losses[0],
        "phys_kfric": phys_col_losses[1],
        "phys_fric": phys_col_losses[2],
        "phys_total": phys_col_losses[3],
    }

    for key in TOLERANCE_METRIC_KEYS:
        loss_dict[key] = tol_metrics[key]

    return total_loss, loss_dict




# =========================================================
# 11 staged weight loading and checkpoint selection
# =========================================================

STAGE_3_CHECKPOINT_FOR_STAGE_4 = "best_mu"

if STAGE_3_CHECKPOINT_FOR_STAGE_4 not in {
    "best_total",
    "best_mu",
}:
    raise ValueError(
        "STAGE_3_CHECKPOINT_FOR_STAGE_4 must be either "
        "'best_total' or 'best_mu'."
    )

best_model_path = os.path.join(
    save_dir,
    f"best_multihead_cnn_stage_{TRAINING_STAGE}.pth",
)

stage_3_best_total_model_path = os.path.join(
    save_dir,
    "best_multihead_cnn_stage_3_best_total.pth",
)

stage_3_best_mu_model_path = os.path.join(
    save_dir,
    "best_multihead_cnn_stage_3_best_mu.pth",
)

if TRAINING_STAGE > 1:
    previous_stage = TRAINING_STAGE - 1

    if TRAINING_STAGE == 4:
        if STAGE_3_CHECKPOINT_FOR_STAGE_4 == "best_total":
            previous_model_path = stage_3_best_total_model_path
        else:
            previous_model_path = stage_3_best_mu_model_path
    else:
        previous_model_path = os.path.join(
            save_dir,
            f"best_multihead_cnn_stage_{previous_stage}.pth",
        )

    if os.path.exists(previous_model_path):
        print(
            f"\nLoading weights from Stage {previous_stage}: "
            f"{previous_model_path}"
        )

        model.load_state_dict(
            torch.load(
                previous_model_path,
                map_location=device,
            )
        )

    else:
        print("\n" + "=" * 80)
        print("WARNING: PREVIOUS-STAGE MODEL NOT FOUND")
        print("=" * 80)
        print(f"Current stage: {TRAINING_STAGE}")
        print("Expected previous model path:")
        print(previous_model_path)

        if TRAINING_STAGE == 4:
            print(
                "\nChange STAGE_3_CHECKPOINT_FOR_STAGE_4 to "
                "'best_total' or 'best_mu', or confirm that the "
                "Stage 3 checkpoint exists."
            )

        print("=" * 80)

else:
    print("\n" + "=" * 80)
    print("STAGE 1: STARTING FROM RANDOM INITIALIZATION")
    print("=" * 80)
    print("No previous-stage model is loaded.")
    print("=" * 80)





# =========================================================
# 12. stage-specific optimizer
# =========================================================
def set_requires_grad(module, flag):
    for param in module.parameters():
        param.requires_grad = flag


def build_stage_optimizer(model, training_stage):
    lr_config = get_stage_learning_rates(training_stage)

    set_requires_grad(model.conv_block, False)
    set_requires_grad(model.shared_fc, False)
    set_requires_grad(model.N0_head, False)
    set_requires_grad(model.mu_head, False)
    set_requires_grad(model.phi_head, False)
    set_requires_grad(model.gamma_head, False)
    set_requires_grad(model.log_lub_head, False)
    set_requires_grad(model.physics_head, False)

    param_groups = []

    if lr_config["backbone"] > 0:
        set_requires_grad(model.conv_block, True)
        set_requires_grad(model.shared_fc, True)

        param_groups.append({
            "name": "backbone",
            "params": (
                list(model.conv_block.parameters())
                + list(model.shared_fc.parameters())
            ),
            "lr": lr_config["backbone"],
        })

    if lr_config["N0"] > 0:
        set_requires_grad(model.N0_head, True)

        param_groups.append({
            "name": "N0_head",
            "params": model.N0_head.parameters(),
            "lr": lr_config["N0"],
        })

    if lr_config["mu"] > 0:
        set_requires_grad(model.mu_head, True)

        param_groups.append({
            "name": "mu_head",
            "params": model.mu_head.parameters(),
            "lr": lr_config["mu"],
        })

    if lr_config["phi"] > 0:
        set_requires_grad(model.phi_head, True)

        param_groups.append({
            "name": "phi_head",
            "params": model.phi_head.parameters(),
            "lr": lr_config["phi"],
        })

    if lr_config["gamma"] > 0:
        set_requires_grad(model.gamma_head, True)

        param_groups.append({
            "name": "gamma_head",
            "params": model.gamma_head.parameters(),
            "lr": lr_config["gamma"],
        })

    if lr_config["lub"] > 0:
        set_requires_grad(model.log_lub_head, True)

        param_groups.append({
            "name": "log_lub_head",
            "params": model.log_lub_head.parameters(),
            "lr": lr_config["lub"],
        })

    if (
        lr_config["physics"] > 0
        and USE_PHYSICS_INFORMED_LOSS_IN_GRADIENT
    ):
        set_requires_grad(model.physics_head, True)

        param_groups.append({
            "name": "physics_head",
            "params": model.physics_head.parameters(),
            "lr": lr_config["physics"],
        })

    if len(param_groups) == 0:
        raise ValueError("No trainable parameter groups were created.")

    optimizer = torch.optim.Adam(
        param_groups,
        weight_decay=weight_decay,
    )

    print("\nStage learning-rate configuration:")
    for key, value in lr_config.items():
        print(f"{key:>10s}: {value:.2e}")

    print("\nTrainable parameter groups:")
    for group in param_groups:
        group_name = group.get("name", "unnamed")
        n_params = sum(p.numel() for p in group["params"])
        print(f"{group_name:>15s}: lr={group['lr']:.2e}, params={n_params}")

    return optimizer


optimizer = build_stage_optimizer(model, TRAINING_STAGE)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=SCHEDULER_PATIENCE,
)




# =========================================================
# 13 training and validation functions
# =========================================================
def run_one_epoch(model, loader, training_stage, optimizer=None):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    totals = {
        "total": 0.0,
        "param": 0.0,

        "parameter_supervision_contribution": 0.0,
        "lubrication_supervision_contribution": 0.0,
        "physics_informed_contribution": 0.0,

        "phys_direct": 0.0,
        "cons_face": 0.0,
        "cons_kfric": 0.0,
        "consistency": 0.0,

        "total_force": 0.0,
        "friction_obs": 0.0,

        "face_target": 0.0,
        "kfric_target": 0.0,
        "bound": 0.0,

        "residual_prior": 0.0,
        "residual_prior_raw": 0.0,
        "phi_prior": 0.0,
        "gamma_prior": 0.0,

        "lub_supervised": 0.0,
        "lub_prior": 0.0,
        "lub_supervised_raw": 0.0,
        "lub_prior_raw": 0.0,
        "lub_mae": 0.0,
        "lub_raw_force": 0.0,
        "lub_raw_force_raw": 0.0,

        "param_N0": 0.0,
        "param_mu": 0.0,
        "param_phi": 0.0,
        "param_gamma": 0.0,

        "phi_mae_deg": 0.0,
        "gamma_mae": 0.0,
        "phi_within_tol": 0.0,
        "gamma_within_tol": 0.0,

        "phys_face": 0.0,
        "phys_kfric": 0.0,
        "phys_fric": 0.0,
        "phys_total": 0.0,
    }

    for key in TOLERANCE_METRIC_KEYS:
        totals[key] = 0.0

    n_samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch in loader:
            (
                x_seq_scaled,
                y_target_scaled,
                p_target_scaled,
                log_lub_target_scaled,
                x_target_raw,
                y_target_raw,
                base_target_raw,
                p_target_raw,
                lub_target_raw,
                a_target_raw,
            ) = batch

            x_seq_scaled = x_seq_scaled.to(device)
            y_target_scaled = y_target_scaled.to(device)
            p_target_scaled = p_target_scaled.to(device)
            log_lub_target_scaled = log_lub_target_scaled.to(device)

            x_target_raw = x_target_raw.to(device)
            y_target_raw = y_target_raw.to(device)
            base_target_raw = base_target_raw.to(device)
            p_target_raw = p_target_raw.to(device)
            lub_target_raw = lub_target_raw.to(device)
            a_target_raw = a_target_raw.to(device)

            if is_train:
                optimizer.zero_grad()

            loss, loss_dict = compute_loss(
                model=model,
                x_seq_scaled=x_seq_scaled,
                y_target_scaled=y_target_scaled,
                base_target_raw=base_target_raw,
                p_target_scaled=p_target_scaled,
                log_lub_target_scaled=log_lub_target_scaled,
                x_target_raw=x_target_raw,
                y_target_raw=y_target_raw,
                p_target_raw=p_target_raw,
                lub_target_raw=lub_target_raw,
                a_target_raw=a_target_raw,
                training_stage=training_stage,
            )

            if is_train:
                if training_stage == 4:
                    loss_for_backward = loss_dict["param_phi"]
                elif training_stage == 5:
                    loss_for_backward = loss_dict["param_gamma"]
                else:
                    loss_for_backward = loss

                loss_for_backward.backward()
                optimizer.step()

            batch_size_now = x_seq_scaled.size(0)
            n_samples += batch_size_now

            for key in totals.keys():
                totals[key] += (
                    loss_dict[key].item()
                    * batch_size_now
                )

    if n_samples == 0:
        raise RuntimeError("No samples were processed.")

    averages = {
        key: value / n_samples
        for key, value in totals.items()
    }

    return averages



# =========================================================
# 14.1 training loop
# =========================================================
history_keys = [
    "total",
    "param",

    "parameter_supervision_contribution",
    "lubrication_supervision_contribution",
    "physics_informed_contribution",

    "phys_direct",
    "cons_face",
    "cons_kfric",
    "consistency",

    "total_force",
    "friction_obs",

    "face_target",
    "kfric_target",
    "bound",

    "residual_prior",
    "residual_prior_raw",
    "phi_prior",
    "gamma_prior",

    "lub_supervised",
    "lub_prior",
    "lub_supervised_raw",
    "lub_prior_raw",
    "lub_raw_force",
    "lub_raw_force_raw",
    "lub_mae",

    "param_N0",
    "param_mu",
    "param_phi",
    "param_gamma",

    "phi_mae_deg",
    "gamma_mae",
    "phi_within_tol",
    "gamma_within_tol",
] + TOLERANCE_METRIC_KEYS + [
    "phys_face",
    "phys_kfric",
    "phys_fric",
    "phys_total",
]

history = {}

for key in history_keys:
    history[f"train_{key}"] = []
    history[f"val_{key}"] = []

best_monitor_value = float("inf")
best_val_total_at_best_model = float("inf")
best_val_param_at_best_model = float("inf")

best_total_value = float("inf")
best_mu_value = float("inf")

best_total_epoch = None
best_mu_epoch = None

epochs_no_improve = 0

epoch_timing_rows = []
stage_training_start_time = timed_now()

best_epoch = None
time_to_best_checkpoint_seconds = np.nan

if TRAINING_STAGE == 1:
    monitor_key = "param"
elif TRAINING_STAGE == 3:
    monitor_key = "total"
elif TRAINING_STAGE == 4:
    monitor_key = "param_phi"
elif TRAINING_STAGE == 5:
    monitor_key = "param_gamma"
else:
    monitor_key = "total"

print("\n" + "=" * 80)
print(f"STARTING {MODEL_VERSION.upper()} TRAINING | STAGE {TRAINING_STAGE}")
print("=" * 80)
print(f"Notebook: {NOTEBOOK_NAME}")
print(f"Model version: {MODEL_VERSION}")

print("Datasets:")
for dataset_path in DATASET_SPECS:
    print(
        f"  {os.path.basename(dataset_path)}: "
        f"{dataset_path}"
    )

print(f"Save directory: {save_dir}")

print(
    "Gradient switches: "
    f"parameter_supervision={USE_PARAMETER_SUPERVISION_IN_GRADIENT}, "
    f"lubrication_supervision={USE_LUBRICATION_SUPERVISION_IN_GRADIENT}, "
    f"physics_informed_loss={USE_PHYSICS_INFORMED_LOSS_IN_GRADIENT}"
)

print(f"Data coverage mode: {DATA_COVERAGE_MODE}")
print(f"PLACEHOLDER_MODE: {PLACEHOLDER_MODE}")

print(
    "CONDITIONAL_PLACEHOLDER_MODE: "
    f"{CONDITIONAL_PLACEHOLDER_MODE}"
)

if CONDITIONAL_PLACEHOLDER_MODE == "true":
    print(
        "Stage 3/4/5 conditional inputs: "
        "true previous targets (teacher forced)"
    )
else:
    print(
        "Stage 3/4/5 conditional inputs: "
        "previous head predictions"
    )

print(f"Input columns: {INPUT_COLS}")
print(f"Auxiliary base targets: {BASE_COLS}")
print("Actual_F_lub is NOT used as input.")

print(
    "True_F_fric_lub is used as an input and represents "
    "Jacking Forces - F_face."
)

print(
    "Architecture: z -> N0/log_F_lub; "
    "[z, N0, log_F_lub] -> mu -> dynamic-residual phi -> "
    "dynamic-residual gamma"
)

print(f"Monitoring validation metric: {monitor_key}")

if TRAINING_STAGE == 3:
    print(
        "Stage 3 best-total checkpoint:"
        f"\n{stage_3_best_total_model_path}"
    )
    print(
        "Stage 3 best-mu checkpoint:"
        f"\n{stage_3_best_mu_model_path}"
    )

print("=" * 80)

for epoch in range(num_epochs):
    epoch_start_time = timed_now()

    train_start_time = timed_now()

    train_metrics = run_one_epoch(
        model=model,
        loader=train_loader,
        training_stage=TRAINING_STAGE,
        optimizer=optimizer,
    )

    train_elapsed_seconds = (
        timed_now() - train_start_time
    )

    val_start_time = timed_now()

    val_metrics = run_one_epoch(
        model=model,
        loader=val_loader,
        training_stage=TRAINING_STAGE,
        optimizer=None,
    )

    val_elapsed_seconds = (
        timed_now() - val_start_time
    )

    epoch_elapsed_seconds = (
        timed_now() - epoch_start_time
    )

    scheduler.step(val_metrics[monitor_key])

    for key in history_keys:
        history[f"train_{key}"].append(
            train_metrics[key]
        )
        history[f"val_{key}"].append(
            val_metrics[key]
        )

    current_monitor_value = val_metrics[monitor_key]
    current_total_value = val_metrics["total"]

    if TRAINING_STAGE == 3:
        # -------------------------------------------------
        # Best total-loss checkpoint
        # -------------------------------------------------
        if current_total_value < best_total_value:
            best_total_value = current_total_value
            best_total_epoch = epoch + 1

            best_monitor_value = current_total_value
            best_val_total_at_best_model = val_metrics["total"]
            best_val_param_at_best_model = val_metrics["param"]
            best_epoch = epoch + 1

            torch.save(
                model.state_dict(),
                stage_3_best_total_model_path,
            )

            # Keep the original generic Stage 3 filename.
            torch.save(
                model.state_dict(),
                best_model_path,
            )

            synchronize_cuda_for_timing()

            time_to_best_checkpoint_seconds = (
                time.perf_counter()
                - stage_training_start_time
            )

            epochs_no_improve = 0

        else:
            epochs_no_improve += 1

        # -------------------------------------------------
        # Best active-parameter checkpoint for Stage 3: μ
        # -------------------------------------------------
        current_mu_value = val_metrics["param_mu"]

        if current_mu_value < best_mu_value:
            best_mu_value = current_mu_value
            best_mu_epoch = epoch + 1

            torch.save(
                model.state_dict(),
                stage_3_best_mu_model_path,
            )

            synchronize_cuda_for_timing()

    else:
        # -------------------------------------------------
        # Stage 1, 2, 4, and 5
        # -------------------------------------------------
        if current_monitor_value < best_monitor_value:
            best_monitor_value = current_monitor_value

            best_val_total_at_best_model = val_metrics["total"]
            best_val_param_at_best_model = val_metrics["param"]

            torch.save(
                model.state_dict(),
                best_model_path,
            )

            synchronize_cuda_for_timing()

            best_epoch = epoch + 1

            time_to_best_checkpoint_seconds = (
                time.perf_counter()
                - stage_training_start_time
            )

            epochs_no_improve = 0

        else:
            epochs_no_improve += 1

    cumulative_elapsed_seconds = (
        timed_now() - stage_training_start_time
    )

    epoch_timing_rows.append({
        "epoch": epoch + 1,
        "epoch_train_time_seconds": train_elapsed_seconds,
        "epoch_validation_time_seconds": val_elapsed_seconds,
        "epoch_wall_time_seconds": epoch_elapsed_seconds,
        "cumulative_training_time_seconds": cumulative_elapsed_seconds,
        "time_to_best_checkpoint_seconds": (
            time_to_best_checkpoint_seconds
        ),
        "best_epoch_so_far": best_epoch,
    })

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"Ep [{epoch + 1:04d}/{num_epochs}] | "
            f"Ttl: {train_metrics['total']:.4f}/"
            f"{val_metrics['total']:.4f} | "
            f"Param: {train_metrics['param']:.4f}/"
            f"{val_metrics['param']:.4f} | "
            f"N0: {train_metrics['param_N0']:.4f}/"
            f"{val_metrics['param_N0']:.4f} | "
            f"mu: {train_metrics['param_mu']:.4f}/"
            f"{val_metrics['param_mu']:.4f} | "
            f"phi: {train_metrics['param_phi']:.4f}/"
            f"{val_metrics['param_phi']:.4f} | "
            f"gamma: {train_metrics['param_gamma']:.4f}/"
            f"{val_metrics['param_gamma']:.4f} | "
            f"TotF: {train_metrics['total_force']:.4f}/"
            f"{val_metrics['total_force']:.4f} | "
            f"FricObs: {train_metrics['friction_obs']:.4f}/"
            f"{val_metrics['friction_obs']:.4f} | "
            f"FaceT: {train_metrics['face_target']:.4f}/"
            f"{val_metrics['face_target']:.4f} | "
            f"KfricT: {train_metrics['kfric_target']:.4f}/"
            f"{val_metrics['kfric_target']:.4f} | "
            f"ResPrior: {train_metrics['residual_prior']:.4f}/"
            f"{val_metrics['residual_prior']:.4f} | "
            f"LubSup: {train_metrics['lub_supervised']:.4f}/"
            f"{val_metrics['lub_supervised']:.4f} | "
            f"LubPrior: {train_metrics['lub_prior']:.4f}/"
            f"{val_metrics['lub_prior']:.4f} | "
            f"LubRaw: {train_metrics['lub_raw_force']:.4f}/"
            f"{val_metrics['lub_raw_force']:.4f} | "
            f"LubMAE: {train_metrics['lub_mae']:.2f}/"
            f"{val_metrics['lub_mae']:.2f} | "
            f"PhiMAE: {train_metrics['phi_mae_deg']:.3f}/"
            f"{val_metrics['phi_mae_deg']:.3f} | "
            f"GamMAE: {train_metrics['gamma_mae']:.3f}/"
            f"{val_metrics['gamma_mae']:.3f}"
        )

    if epochs_no_improve >= PATIENCE:
        print(
            f"Early stopping at epoch {epoch + 1}"
        )
        break


synchronize_cuda_for_timing()

stage_training_time_seconds = (
    time.perf_counter()
    - stage_training_start_time
)

epochs_completed = len(epoch_timing_rows)

if epochs_completed == 0:
    raise RuntimeError(
        "No training epochs were completed."
    )

epoch_timing_df = pd.DataFrame(epoch_timing_rows)

if best_epoch is None:
    best_epoch = int(
        np.argmin(
            history[f"val_{monitor_key}"]
        ) + 1
    )

    time_to_best_checkpoint_seconds = float(
        epoch_timing_df.loc[
            epoch_timing_df["epoch"] == best_epoch,
            "cumulative_training_time_seconds",
        ].iloc[0]
    )

history_df = pd.DataFrame(history)

history_df.insert(
    0,
    "epoch",
    np.arange(1, len(history_df) + 1),
)

history_df = history_df.merge(
    epoch_timing_df,
    on="epoch",
    how="left",
)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

average_epoch_time_seconds = float(
    epoch_timing_df["epoch_wall_time_seconds"].mean()
)

average_train_epoch_time_seconds = float(
    epoch_timing_df["epoch_train_time_seconds"].mean()
)

average_validation_epoch_time_seconds = float(
    epoch_timing_df["epoch_validation_time_seconds"].mean()
)

timing_summary_df = pd.DataFrame([{
    "architecture": ARCHITECTURE_NAME,
    "model_version": MODEL_VERSION,
    "version": NOTEBOOK_NAME,
    "trained_stage": TRAINING_STAGE,
    "use_parameter_supervision": (
        USE_PARAMETER_SUPERVISION_IN_GRADIENT
    ),
    "use_lubrication_supervision": (
        USE_LUBRICATION_SUPERVISION_IN_GRADIENT
    ),
    "use_physics_informed_loss": (
        USE_PHYSICS_INFORMED_LOSS_IN_GRADIENT
    ),
    "device": str(device),
    "dataset_specs": " | ".join(DATASET_SPECS),
    "data_coverage_mode": DATA_COVERAGE_MODE,
    "window_size": window_size,
    "batch_size": batch_size,
    "max_window_size_for_comparison": MAX_WINDOW_SIZE,
    "common_target_start_position": COMMON_TARGET_START_POS,
    "dropout": dropout,
    "num_epochs_requested": num_epochs,
    "epochs_completed": epochs_completed,
    "monitor_key": monitor_key,
    "best_epoch": best_epoch,
    "best_monitor_value": best_monitor_value,
    "best_val_total": best_val_total_at_best_model,
    "best_val_param": best_val_param_at_best_model,
    "best_total_epoch": best_total_epoch,
    "best_total_validation_loss": (
        best_total_value
        if np.isfinite(best_total_value)
        else np.nan
    ),
    "best_mu_epoch": best_mu_epoch,
    "best_mu_validation_loss": (
        best_mu_value
        if np.isfinite(best_mu_value)
        else np.nan
    ),
    "best_total_checkpoint_path": (
        stage_3_best_total_model_path
        if TRAINING_STAGE == 3
        else ""
    ),
    "best_mu_checkpoint_path": (
        stage_3_best_mu_model_path
        if TRAINING_STAGE == 3
        else ""
    ),
    "training_time_seconds": stage_training_time_seconds,
    "training_time_minutes": (
        stage_training_time_seconds / 60.0
    ),
    "average_epoch_time_seconds": (
        average_epoch_time_seconds
    ),
    "average_train_epoch_time_seconds": (
        average_train_epoch_time_seconds
    ),
    "average_validation_epoch_time_seconds": (
        average_validation_epoch_time_seconds
    ),
    "average_seconds_per_train_window": (
        average_train_epoch_time_seconds
        / max(len(train_dataset), 1)
    ),
    "average_train_windows_per_second": (
        len(train_dataset)
        / max(
            average_train_epoch_time_seconds,
            1e-12,
        )
    ),
    "time_to_best_checkpoint_seconds": (
        time_to_best_checkpoint_seconds
    ),
    "time_to_best_checkpoint_minutes": (
        time_to_best_checkpoint_seconds / 60.0
    ),
    "total_parameters": total_parameters,
    "trainable_parameters_at_stage_end": (
        trainable_parameters
    ),
    "num_train_windows": len(train_dataset),
    "num_validation_windows": len(val_dataset),
    "num_test_windows": len(test_dataset),
}])

training_history_path = os.path.join(
    save_dir,
    f"training_history_stage_{TRAINING_STAGE}.xlsx",
)

with pd.ExcelWriter(
    training_history_path,
    engine="openpyxl",
) as writer:

    history_df.to_excel(
        writer,
        sheet_name="history",
        index=False,
    )

    timing_summary_df.to_excel(
        writer,
        sheet_name="timing_summary",
        index=False,
    )

    epoch_timing_df.to_excel(
        writer,
        sheet_name="epoch_timing",
        index=False,
    )


def evaluate_and_print_checkpoint(
    checkpoint_label,
    checkpoint_path,
):
    print("\n" + "=" * 80)
    print(checkpoint_label)
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")

    checkpoint_state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint_state)

    train_checkpoint_metrics = run_one_epoch(
        model=model,
        loader=train_loader,
        training_stage=TRAINING_STAGE,
        optimizer=None,
    )

    val_checkpoint_metrics = run_one_epoch(
        model=model,
        loader=val_loader,
        training_stage=TRAINING_STAGE,
        optimizer=None,
    )

    test_checkpoint_metrics = run_one_epoch(
        model=model,
        loader=test_loader,
        training_stage=TRAINING_STAGE,
        optimizer=None,
    )

    def metric_triplet(
        metric_name,
        decimals=4,
    ):
        return (
            f"{train_checkpoint_metrics[metric_name]:.{decimals}f}/"
            f"{val_checkpoint_metrics[metric_name]:.{decimals}f}/"
            f"{test_checkpoint_metrics[metric_name]:.{decimals}f}"
        )

    print(
        f"Ttl: {metric_triplet('total')} | "
        f"Param: {metric_triplet('param')} | "
        f"N0: {metric_triplet('param_N0')} | "
        f"mu: {metric_triplet('param_mu')} | "
        f"phi: {metric_triplet('param_phi')} | "
        f"gamma: {metric_triplet('param_gamma')} | "
        f"TotF: {metric_triplet('total_force')} | "
        f"FricObs: {metric_triplet('friction_obs')} | "
        f"FaceT: {metric_triplet('face_target')} | "
        f"KfricT: {metric_triplet('kfric_target')} | "
        f"ResPrior: {metric_triplet('residual_prior')} | "
        f"LubSup: {metric_triplet('lub_supervised')} | "
        f"LubPrior: {metric_triplet('lub_prior')} | "
        f"LubRaw: {metric_triplet('lub_raw_force')} | "
        f"LubMAE: {metric_triplet('lub_mae', 2)} | "
        f"PhiMAE: {metric_triplet('phi_mae_deg', 3)} | "
        f"GamMAE: {metric_triplet('gamma_mae', 3)}"
    )


print("\nTraining history saved to:")
print(training_history_path)

print("\nTraining timing summary:")

# UPDATED: print each key and value on the same line.
timing_summary_record = timing_summary_df.iloc[0].to_dict()

timing_label_width = max(
    len(str(key))
    for key in timing_summary_record
) + 4

for key, value in timing_summary_record.items():
    print(
        f"{str(key):<{timing_label_width}} {value}"
    )

print("\nCheckpoint files:")
print(f"Generic checkpoint: {best_model_path}")


if TRAINING_STAGE == 3:
    print("\n" + "=" * 80)
    print("STAGE 3 CHECKPOINT COMPARISON")
    print("=" * 80)

    # First print the complete best-total checkpoint result.
    evaluate_and_print_checkpoint(
        checkpoint_label=(
            "STAGE 3 — BEST TOTAL-LOSS CHECKPOINT"
        ),
        checkpoint_path=stage_3_best_total_model_path,
    )

    # Then print the complete best-mu checkpoint result.
    evaluate_and_print_checkpoint(
        checkpoint_label=(
            "STAGE 3 — BEST MU CHECKPOINT"
        ),
        checkpoint_path=stage_3_best_mu_model_path,
    )

    print(
        "\nBest-total epoch: "
        f"{best_total_epoch}"
    )

    print(
        "Best-total validation loss: "
        f"{best_total_value:.6f}"
    )

    print(
        "Best-total checkpoint: "
        f"{stage_3_best_total_model_path}"
    )

    print(
        "\nBest-mu epoch: "
        f"{best_mu_epoch}"
    )

    print(
        "Best-mu validation loss: "
        f"{best_mu_value:.6f}"
    )

    print(
        "Best-mu checkpoint: "
        f"{stage_3_best_mu_model_path}"
    )

else:
    evaluate_and_print_checkpoint(
        checkpoint_label=(
            f"STAGE {TRAINING_STAGE} — BEST CHECKPOINT"
        ),
        checkpoint_path=best_model_path,
    )

print("\n" + "=" * 80)
print(
    f"STAGE {TRAINING_STAGE} TRAINING COMPLETE"
)
print("=" * 80)



# =========================================================
# 15 evaluate all trained stages and save one Excel output
# =========================================================

import os
import numpy as np
import pandas as pd
import torch

from IPython.display import display


# Change this list based on which NN stages you have already trained.
# Examples:
# STAGES_TO_EVALUATE = [1]
# STAGES_TO_EVALUATE = [1, 2, 3]
# STAGES_TO_EVALUATE = [1, 2, 3, 4, 5]
# STAGES_TO_EVALUATE = [1, 2, 3, 4, 5, 6]

STAGES_TO_EVALUATE = [1, 2, 3, 4, 5]



EVALUATE_WITH_OWN_STAGE_SETTING = True


# If you want to evaluate all checkpoints using one common stage setting,
# set EVALUATE_WITH_OWN_STAGE_SETTING = False.
COMMON_EVAL_STAGE = 5


all_stage_excel_path = os.path.join(
    save_dir,
    "surrogate_cnn_based_assessment_all_stage_training_validation_test_summary.xlsx",
)


def get_stage_checkpoint_path(stage):


    return os.path.join(
        save_dir,
        f"best_multihead_cnn_stage_{stage}.pth",
    )


def get_stage_history_path(stage):


    return os.path.join(
        save_dir,
        f"training_history_stage_{stage}.xlsx",
    )


MAPE_EPS = 1e-8


def safe_mape_percent(pred, target, eps=MAPE_EPS):


    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)

    denominator = np.maximum(np.abs(target), eps)
    ape = np.abs((pred - target) / denominator) * 100.0

    return float(np.mean(ape))


def safe_r2_score(pred, target, eps=1e-12):


    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)

    ss_res = np.sum((pred - target) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)

    if ss_tot < eps:
        return np.nan

    return float(1.0 - ss_res / ss_tot)


def safe_corr(pred, target, eps=1e-12):

    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)

    if len(pred) < 2:
        return np.nan

    if np.std(pred) <= eps or np.std(target) <= eps:
        return np.nan

    return float(np.corrcoef(pred, target)[0, 1])


def add_prediction_metrics_to_row(row, prefix, pred, target):


    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)

    error = pred - target
    abs_error = np.abs(error)
    sq_error = error ** 2

    row[f"{prefix}_mae_raw"] = float(np.mean(abs_error))
    row[f"{prefix}_mse_raw"] = float(np.mean(sq_error))
    row[f"{prefix}_rmse_raw"] = float(np.sqrt(np.mean(sq_error)))
    row[f"{prefix}_bias_raw"] = float(np.mean(error))

    row[f"{prefix}_mape_percent"] = safe_mape_percent(
        pred=pred,
        target=target,
    )

    row[f"{prefix}_r2"] = safe_r2_score(
        pred=pred,
        target=target,
    )

    row[f"{prefix}_corr"] = safe_corr(
        pred=pred,
        target=target,
    )

    return row


def add_scaled_error_metrics_to_row(row, prefix, pred_scaled, target_scaled):

    pred_scaled = np.asarray(pred_scaled, dtype=np.float64).reshape(-1)
    target_scaled = np.asarray(target_scaled, dtype=np.float64).reshape(-1)

    error_scaled = pred_scaled - target_scaled
    abs_error_scaled = np.abs(error_scaled)
    sq_error_scaled = error_scaled ** 2

    row[f"{prefix}_mae_scaled"] = float(np.mean(abs_error_scaled))
    row[f"{prefix}_mse_scaled"] = float(np.mean(sq_error_scaled))
    row[f"{prefix}_rmse_scaled"] = float(np.sqrt(np.mean(sq_error_scaled)))
    row[f"{prefix}_bias_scaled"] = float(np.mean(error_scaled))

    return row


def load_stage_checkpoint(model, stage, device):
    checkpoint_path = get_stage_checkpoint_path(stage)

    if not os.path.exists(checkpoint_path):
        print("\n" + "=" * 100)
        print(f"WARNING: NN Stage {stage} checkpoint not found.")
        print("Expected checkpoint path:")
        print(checkpoint_path)
        print("This stage will be skipped.")
        print("=" * 100)
        return None, None

    print("\n" + "=" * 100)
    print(f"Loading NN Stage {stage} checkpoint")
    print(checkpoint_path)
    print("=" * 100)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, checkpoint_path



def collect_prediction_metrics_for_loader(
    model,
    loader,
    eval_stage_setting,
    device,
):


    model.eval()

    all_params_pred_raw = []
    all_params_true_raw = []

    all_params_pred_scaled = []
    all_params_true_scaled = []

    all_physics_pred_raw = []
    all_physics_true_raw = []

    all_physics_pred_scaled = []
    all_physics_true_scaled = []

    all_log_lub_pred_scaled = []
    all_log_lub_true_scaled = []

    all_log_lub_pred_raw = []
    all_log_lub_true_raw = []

    all_F_lub_pred_raw = []
    all_F_lub_true_raw = []

    inference_start_time = timed_now()

    detach_previous_eval = eval_stage_setting <= 5

    with torch.no_grad():
        for batch in loader:
            (
                x_seq_scaled,
                y_target_scaled,
                p_target_scaled,
                log_lub_target_scaled,
                x_target_raw,
                y_target_raw,
                base_target_raw,
                p_target_raw,
                lub_target_raw,
                a_target_raw,
            ) = batch

            x_seq_scaled = x_seq_scaled.to(device)
            y_target_scaled = y_target_scaled.to(device)
            p_target_scaled = p_target_scaled.to(device)
            log_lub_target_scaled = log_lub_target_scaled.to(device)
            base_target_raw = base_target_raw.to(device)
            lub_target_raw = lub_target_raw.to(device)

            (
                params_pred_scaled,
                physics_pred_scaled,
                log_lub_pred_scaled,
                log_lub_pred_raw,
                F_lub_pred_raw,
            ) = model(
                x_seq_scaled,
                detach_previous=detach_previous_eval,
                # Version 4.1.10 requires drive-level bases for the
                # bounded residual phi/gamma heads.
                base_target_raw=base_target_raw,
                # Required for teacher-forced conditional stages.
                y_target_scaled=y_target_scaled,
                log_lub_target_scaled=log_lub_target_scaled,
                training_stage=eval_stage_setting,
                conditional_placeholder_mode=(
                    globals().get(
                        "CONDITIONAL_PLACEHOLDER_MODE",
                        "predicted",
                    )
                ),
            )

            params_pred_raw = inverse_transform_torch(
                params_pred_scaled,
                scaler_Y,
                device,
            )

            physics_pred_raw = inverse_transform_torch(
                physics_pred_scaled,
                scaler_P,
                device,
            )

            log_lub_target_raw = torch.log(
                torch.clamp(
                    lub_target_raw,
                    min=LUB_EPS,
                )
            )

            all_params_pred_raw.append(params_pred_raw.detach().cpu().numpy())
            all_params_true_raw.append(y_target_raw.detach().cpu().numpy())

            all_params_pred_scaled.append(params_pred_scaled.detach().cpu().numpy())
            all_params_true_scaled.append(y_target_scaled.detach().cpu().numpy())

            all_physics_pred_raw.append(physics_pred_raw.detach().cpu().numpy())
            all_physics_true_raw.append(p_target_raw.detach().cpu().numpy())

            all_physics_pred_scaled.append(physics_pred_scaled.detach().cpu().numpy())
            all_physics_true_scaled.append(p_target_scaled.detach().cpu().numpy())

            all_log_lub_pred_scaled.append(log_lub_pred_scaled.detach().cpu().numpy())
            all_log_lub_true_scaled.append(log_lub_target_scaled.detach().cpu().numpy())

            all_log_lub_pred_raw.append(log_lub_pred_raw.detach().cpu().numpy())
            all_log_lub_true_raw.append(log_lub_target_raw.detach().cpu().numpy())

            all_F_lub_pred_raw.append(F_lub_pred_raw.detach().cpu().numpy())
            all_F_lub_true_raw.append(lub_target_raw.detach().cpu().numpy())

    params_pred_raw = np.vstack(all_params_pred_raw)
    params_true_raw = np.vstack(all_params_true_raw)

    params_pred_scaled = np.vstack(all_params_pred_scaled)
    params_true_scaled = np.vstack(all_params_true_scaled)

    physics_pred_raw = np.vstack(all_physics_pred_raw)
    physics_true_raw = np.vstack(all_physics_true_raw)

    physics_pred_scaled = np.vstack(all_physics_pred_scaled)
    physics_true_scaled = np.vstack(all_physics_true_scaled)

    log_lub_pred_scaled = np.vstack(all_log_lub_pred_scaled)
    log_lub_true_scaled = np.vstack(all_log_lub_true_scaled)

    log_lub_pred_raw = np.vstack(all_log_lub_pred_raw)
    log_lub_true_raw = np.vstack(all_log_lub_true_raw)

    F_lub_pred_raw = np.vstack(all_F_lub_pred_raw)
    F_lub_true_raw = np.vstack(all_F_lub_true_raw)

    inference_time_seconds = timed_now() - inference_start_time
    n_inference_windows = len(params_pred_raw)

    pred_metric_row = {}

    for i, col in enumerate(PARAM_COLS):
        pred_metric_row = add_prediction_metrics_to_row(
            row=pred_metric_row,
            prefix=f"{col}",
            pred=params_pred_raw[:, i],
            target=params_true_raw[:, i],
        )

        pred_metric_row = add_scaled_error_metrics_to_row(
            row=pred_metric_row,
            prefix=f"{col}",
            pred_scaled=params_pred_scaled[:, i],
            target_scaled=params_true_scaled[:, i],
        )


    for i, col in enumerate(PHYSICS_COLS):
        pred_metric_row = add_prediction_metrics_to_row(
            row=pred_metric_row,
            prefix=f"{col}_physics",
            pred=physics_pred_raw[:, i],
            target=physics_true_raw[:, i],
        )

        pred_metric_row = add_scaled_error_metrics_to_row(
            row=pred_metric_row,
            prefix=f"{col}_physics",
            pred_scaled=physics_pred_scaled[:, i],
            target_scaled=physics_true_scaled[:, i],
        )


    pred_metric_row = add_prediction_metrics_to_row(
        row=pred_metric_row,
        prefix="Actual_F_lub",
        pred=F_lub_pred_raw[:, 0],
        target=F_lub_true_raw[:, 0],
    )


    pred_metric_row = add_prediction_metrics_to_row(
        row=pred_metric_row,
        prefix="log_F_lub",
        pred=log_lub_pred_raw[:, 0],
        target=log_lub_true_raw[:, 0],
    )

    pred_metric_row = add_prediction_metrics_to_row(
        row=pred_metric_row,
        prefix="log_F_lub_scaled",
        pred=log_lub_pred_scaled[:, 0],
        target=log_lub_true_scaled[:, 0],
    )

    pred_metric_row = add_scaled_error_metrics_to_row(
        row=pred_metric_row,
        prefix="log_F_lub",
        pred_scaled=log_lub_pred_scaled[:, 0],
        target_scaled=log_lub_true_scaled[:, 0],
    )

    pred_metric_row.update({
        "inference_time_seconds": inference_time_seconds,
        "inference_windows": n_inference_windows,
        "inference_ms_per_window": (
            1000.0 * inference_time_seconds
            / max(n_inference_windows, 1)
        ),
        "inference_windows_per_second": (
            n_inference_windows
            / max(inference_time_seconds, 1e-12)
        ),
    })

    return pred_metric_row


def evaluate_stage_checkpoint(model, stage, device):
    model, checkpoint_path = load_stage_checkpoint(
        model=model,
        stage=stage,
        device=device,
    )

    if model is None:
        return []

    if EVALUATE_WITH_OWN_STAGE_SETTING:
        eval_stage_setting = stage
    else:
        eval_stage_setting = COMMON_EVAL_STAGE

    loaders_dict = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    rows = []

    for split_name, loader in loaders_dict.items():
        print(
            f"\nEvaluating NN Stage {stage} checkpoint on {split_name} set "
            f"using training_stage={eval_stage_setting}..."
        )

        loss_metrics = run_one_epoch(
            model=model,
            loader=loader,
            training_stage=eval_stage_setting,
            optimizer=None,
        )

        prediction_metrics = collect_prediction_metrics_for_loader(
            model=model,
            loader=loader,
            eval_stage_setting=eval_stage_setting,
            device=device,
        )

        row = {
            "architecture": ARCHITECTURE_NAME,
            "version": MODEL_VERSION,
            "window_size": window_size,
            "batch_size": batch_size,
            "trained_stage": stage,
            "eval_stage_setting": eval_stage_setting,
            "split": split_name,
            "checkpoint_path": checkpoint_path,
        }

        for key, value in loss_metrics.items():
            row[key] = value

        for key, value in prediction_metrics.items():
            row[key] = value

        rows.append(row)

        print("-" * 100)
        print(f"NN Stage {stage} | {split_name} metrics")
        print("-" * 100)

        important_keys = [
            "total",
            "param",

            "phys_direct",
            "cons_face",
            "cons_kfric",
            "consistency",

            "total_force",
            "friction_obs",

            "face_target",
            "kfric_target",
            "bound",

            "residual_prior",
            "phi_prior",
            "gamma_prior",

            "lub_supervised",
            "lub_prior",
            "lub_supervised_raw",
            "lub_prior_raw",
            "lub_mae",

            "param_N0",
            "param_mu",
            "param_phi",
            "param_gamma",

            "phi_mae_deg",
            "gamma_mae",
            "phi_within_tol",
            "gamma_within_tol",

            "phys_face",
            "phys_kfric",
            "phys_fric",
            "phys_total",

            "N0_mae_raw",
            "N0_mse_raw",
            "N0_rmse_raw",
            "N0_mse_scaled",
            "N0_rmse_scaled",
            "N0_mape_percent",
            "N0_r2",
            "N0_corr",

            "mu_mae_raw",
            "mu_mse_raw",
            "mu_rmse_raw",
            "mu_mse_scaled",
            "mu_rmse_scaled",
            "mu_mape_percent",
            "mu_r2",
            "mu_corr",

            "phi_deg_mae_raw",
            "phi_deg_mse_raw",
            "phi_deg_rmse_raw",
            "phi_deg_mse_scaled",
            "phi_deg_rmse_scaled",
            "phi_deg_mape_percent",
            "phi_deg_r2",
            "phi_deg_corr",

            "gamma_mae_raw",
            "gamma_mse_raw",
            "gamma_rmse_raw",
            "gamma_mse_scaled",
            "gamma_rmse_scaled",
            "gamma_mape_percent",
            "gamma_r2",
            "gamma_corr",

            "Actual_F_lub_mae_raw",
            "Actual_F_lub_mse_raw",
            "Actual_F_lub_rmse_raw",
            "Actual_F_lub_mape_percent",
            "Actual_F_lub_r2",
            "Actual_F_lub_corr",

            "log_F_lub_mse_raw",
            "log_F_lub_rmse_raw",
            "log_F_lub_mse_scaled",
            "log_F_lub_rmse_scaled",
        ]

        for key in important_keys:
            if key in row:
                print(f"{key:>35s}: {row[key]:.6f}")

    return rows



def add_lubrication_derived_columns(df_in):

    df_out = df_in.copy()

    log_lub_scale = float(scaler_L.scale_[0])

    try:
        prior_log_sigma = float(SIGMA_LOG_FLUB_PRIOR)
    except Exception:
        prior_log_sigma = float(sigma_log_flub_prior_tensor.detach().cpu().item())

    if not np.isfinite(prior_log_sigma) or prior_log_sigma <= 0:
        prior_log_sigma = 1.0

    # For long-format columns.
    if "lub_supervised_raw" in df_out.columns:
        df_out["lub_supervised_raw_rmse_scaled_log"] = np.sqrt(
            np.maximum(df_out["lub_supervised_raw"].astype(float), 0.0)
        )

        df_out["lub_supervised_raw_rmse_log_unit"] = (
            df_out["lub_supervised_raw_rmse_scaled_log"] * log_lub_scale
        )

    if "lub_prior_raw" in df_out.columns:
        df_out["lub_prior_raw_rmse_standardized_log"] = np.sqrt(
            np.maximum(df_out["lub_prior_raw"].astype(float), 0.0)
        )

        df_out["lub_prior_raw_rmse_log_unit"] = (
            df_out["lub_prior_raw_rmse_standardized_log"] * prior_log_sigma
        )

    if "lub_supervised" in df_out.columns:
        df_out["total_contribution_lub_supervised"] = (
            lambda_lub_supervised * df_out["lub_supervised"].astype(float)
        )

    if "lub_prior" in df_out.columns:
        df_out["total_contribution_lub_prior"] = (
            lambda_lub_prior * df_out["lub_prior"].astype(float)
        )

    # For wide-format columns.
    for split in ["train", "val", "test"]:
        sup_raw_col = f"lub_supervised_raw_{split}"
        prior_raw_col = f"lub_prior_raw_{split}"
        sup_col = f"lub_supervised_{split}"
        prior_col = f"lub_prior_{split}"

        if sup_raw_col in df_out.columns:
            df_out[f"lub_supervised_raw_rmse_scaled_log_{split}"] = np.sqrt(
                np.maximum(df_out[sup_raw_col].astype(float), 0.0)
            )

            df_out[f"lub_supervised_raw_rmse_log_unit_{split}"] = (
                df_out[f"lub_supervised_raw_rmse_scaled_log_{split}"] * log_lub_scale
            )

        if prior_raw_col in df_out.columns:
            df_out[f"lub_prior_raw_rmse_standardized_log_{split}"] = np.sqrt(
                np.maximum(df_out[prior_raw_col].astype(float), 0.0)
            )

            df_out[f"lub_prior_raw_rmse_log_unit_{split}"] = (
                df_out[f"lub_prior_raw_rmse_standardized_log_{split}"] * prior_log_sigma
            )

        if sup_col in df_out.columns:
            df_out[f"total_contribution_lub_supervised_{split}"] = (
                lambda_lub_supervised * df_out[sup_col].astype(float)
            )

        if prior_col in df_out.columns:
            df_out[f"total_contribution_lub_prior_{split}"] = (
                lambda_lub_prior * df_out[prior_col].astype(float)
            )

    # For training-history columns.
    for split in ["train", "val"]:
        sup_raw_col = f"{split}_lub_supervised_raw"
        prior_raw_col = f"{split}_lub_prior_raw"
        sup_col = f"{split}_lub_supervised"
        prior_col = f"{split}_lub_prior"

        if sup_raw_col in df_out.columns:
            df_out[f"{split}_lub_supervised_raw_rmse_scaled_log"] = np.sqrt(
                np.maximum(df_out[sup_raw_col].astype(float), 0.0)
            )

            df_out[f"{split}_lub_supervised_raw_rmse_log_unit"] = (
                df_out[f"{split}_lub_supervised_raw_rmse_scaled_log"] * log_lub_scale
            )

        if prior_raw_col in df_out.columns:
            df_out[f"{split}_lub_prior_raw_rmse_standardized_log"] = np.sqrt(
                np.maximum(df_out[prior_raw_col].astype(float), 0.0)
            )

            df_out[f"{split}_lub_prior_raw_rmse_log_unit"] = (
                df_out[f"{split}_lub_prior_raw_rmse_standardized_log"] * prior_log_sigma
            )

        if sup_col in df_out.columns:
            df_out[f"{split}_total_contribution_lub_supervised"] = (
                lambda_lub_supervised * df_out[sup_col].astype(float)
            )

        if prior_col in df_out.columns:
            df_out[f"{split}_total_contribution_lub_prior"] = (
                lambda_lub_prior * df_out[prior_col].astype(float)
            )

    return df_out.copy()



def collect_all_stage_histories(stages):
    history_dfs = []

    for stage in stages:
        history_path = get_stage_history_path(stage)

        if not os.path.exists(history_path):
            print(f"\nNo NN training history file found for Stage {stage}:")
            print(history_path)
            continue

        history_df = pd.read_excel(history_path)

        history_df = history_df.copy()
        history_df = history_df.loc[:, ~history_df.columns.duplicated()].copy()

        if "epoch" not in history_df.columns:
            history_df["epoch"] = np.arange(1, len(history_df) + 1)

        history_df["trained_stage"] = stage
        history_df["history_path"] = history_path

        front_cols = ["trained_stage", "history_path", "epoch"]
        other_cols = [
            col for col in history_df.columns
            if col not in front_cols
        ]

        history_df = history_df[front_cols + other_cols].copy()
        history_df = add_lubrication_derived_columns(history_df)

        history_dfs.append(history_df)

    if len(history_dfs) == 0:
        return pd.DataFrame()

    return pd.concat(history_dfs, ignore_index=True)


def collect_training_timing_summaries(stages):
    """Read the per-stage timing sheets saved by the training block."""
    timing_dfs = []

    for stage in stages:
        history_path = get_stage_history_path(stage)

        if not os.path.exists(history_path):
            continue

        try:
            timing_df = pd.read_excel(
                history_path,
                sheet_name="timing_summary",
            )
        except ValueError:
            print(
                f"No timing_summary sheet found for NN Stage {stage}:"
            )
            continue

        timing_df = timing_df.copy()
        timing_df = timing_df.loc[
            :, ~timing_df.columns.duplicated()
        ].copy()
        timing_df["trained_stage"] = stage
        timing_df["timing_path"] = history_path
        timing_dfs.append(timing_df)

    if len(timing_dfs) == 0:
        return pd.DataFrame()

    return pd.concat(timing_dfs, ignore_index=True)



def summarize_best_epoch_from_history(history_all_df):
    if len(history_all_df) == 0:
        return pd.DataFrame()

    best_rows = []

    for stage in sorted(history_all_df["trained_stage"].unique()):
        stage_hist = history_all_df[
            history_all_df["trained_stage"] == stage
        ].copy()

        if len(stage_hist) == 0:
            continue

        if stage == 1 and "val_param" in stage_hist.columns:
            monitor_col = "val_param"
        elif "val_total" in stage_hist.columns:
            monitor_col = "val_total"
        else:
            print(
                f"NN Stage {stage}: cannot find val_param or val_total "
                "for best-epoch summary."
            )
            continue

        best_idx = stage_hist[monitor_col].idxmin()
        best_row = stage_hist.loc[best_idx].copy()

        best_row["monitor_col"] = monitor_col
        best_row["best_monitor_value_from_history"] = best_row[monitor_col]

        best_rows.append(best_row)

    if len(best_rows) == 0:
        return pd.DataFrame()

    return pd.DataFrame(best_rows).reset_index(drop=True)



print("\n" + "=" * 120)
print(f"RUNNING {MODEL_VERSION.upper()} ALL-STAGE EVALUATION")
print("=" * 120)
print(f"save_dir: {save_dir}")
print("Expected checkpoint pattern:")
print(os.path.join(save_dir, "best_multihead_cnn_stage_{stage}.pth"))
print("=" * 120)

all_eval_rows = []
missing_stages = []

for stage in STAGES_TO_EVALUATE:
    checkpoint_path = get_stage_checkpoint_path(stage)

    if not os.path.exists(checkpoint_path):
        missing_stages.append(stage)

    stage_rows = evaluate_stage_checkpoint(
        model=model,
        stage=stage,
        device=device,
    )

    all_eval_rows.extend(stage_rows)


if len(all_eval_rows) == 0:
    raise RuntimeError(
        "No NN stages were evaluated. Please check whether the NN checkpoint files exist."
    )

metrics_long_df = pd.DataFrame(all_eval_rows)
metrics_long_df = add_lubrication_derived_columns(metrics_long_df)



metadata_cols = [
    "architecture",
    "version",
    "window_size",
    "batch_size",
    "trained_stage",
    "eval_stage_setting",
    "split",
    "checkpoint_path",
]

metric_cols = [
    col for col in metrics_long_df.columns
    if col not in metadata_cols
]

metrics_wide_df = metrics_long_df.pivot_table(
    index=[
        "architecture",
        "version",
        "window_size",
        "batch_size",
        "trained_stage",
        "eval_stage_setting",
        "checkpoint_path",
    ],
    columns="split",
    values=metric_cols,
    aggfunc="first",
)

metrics_wide_df.columns = [
    f"{metric}_{split}" for metric, split in metrics_wide_df.columns
]

metrics_wide_df = metrics_wide_df.reset_index()
metrics_wide_df = add_lubrication_derived_columns(metrics_wide_df)


loss_base_names = [
    "total",
    "param",

    "parameter_supervision_contribution",
    "lubrication_supervision_contribution",
    "physics_informed_contribution",

    "phys_direct",
    "cons_face",
    "cons_kfric",
    "consistency",

    "total_force",
    "friction_obs",

    "face_target",
    "kfric_target",
    "bound",

    "residual_prior",
    "phi_prior",
    "gamma_prior",

    "lub_supervised",
    "lub_prior",
    "lub_supervised_raw",
    "lub_prior_raw",

    "lub_supervised_raw_rmse_scaled_log",
    "lub_supervised_raw_rmse_log_unit",
    "lub_prior_raw_rmse_standardized_log",
    "lub_prior_raw_rmse_log_unit",

    "total_contribution_lub_supervised",
    "total_contribution_lub_prior",

    "param_N0",
    "param_mu",
    "param_phi",
    "param_gamma",

    "phys_face",
    "phys_kfric",
    "phys_fric",
    "phys_total",
]

loss_summary_cols = [
    "trained_stage",
    "eval_stage_setting",
]

for metric_name in loss_base_names:
    for split in ["train", "val", "test"]:
        col = f"{metric_name}_{split}"

        if col in metrics_wide_df.columns:
            loss_summary_cols.append(col)

loss_summary_df = metrics_wide_df[loss_summary_cols].copy()



prediction_base_names = []

prediction_prefixes_raw = []

for col in PARAM_COLS:
    prediction_prefixes_raw.append(col)

for col in PHYSICS_COLS:
    prediction_prefixes_raw.append(f"{col}_physics")

prediction_prefixes_raw += [
    "Actual_F_lub",
    "log_F_lub",
    "log_F_lub_scaled",
]

for prefix in prediction_prefixes_raw:
    prediction_base_names += [
        f"{prefix}_mae_raw",
        f"{prefix}_mse_raw",
        f"{prefix}_rmse_raw",
        f"{prefix}_bias_raw",
        f"{prefix}_mape_percent",
        f"{prefix}_r2",
        f"{prefix}_corr",
    ]

prediction_prefixes_scaled = []

for col in PARAM_COLS:
    prediction_prefixes_scaled.append(col)

for col in PHYSICS_COLS:
    prediction_prefixes_scaled.append(f"{col}_physics")

prediction_prefixes_scaled += [
    "log_F_lub",
]

for prefix in prediction_prefixes_scaled:
    prediction_base_names += [
        f"{prefix}_mae_scaled",
        f"{prefix}_mse_scaled",
        f"{prefix}_rmse_scaled",
        f"{prefix}_bias_scaled",
    ]

prediction_summary_cols = [
    "trained_stage",
    "eval_stage_setting",
]

for metric_name in prediction_base_names:
    for split in ["train", "val", "test"]:
        col = f"{metric_name}_{split}"

        if col in metrics_wide_df.columns:
            prediction_summary_cols.append(col)

prediction_summary_df = metrics_wide_df[prediction_summary_cols].copy()



compact_cols = [
    "architecture",
    "version",
    "window_size",
    "batch_size",
    "trained_stage",
    "eval_stage_setting",

    "total_train",
    "total_val",
    "total_test",

    "param_train",
    "param_val",
    "param_test",

    "parameter_supervision_contribution_train",
    "parameter_supervision_contribution_val",
    "parameter_supervision_contribution_test",

    "lubrication_supervision_contribution_train",
    "lubrication_supervision_contribution_val",
    "lubrication_supervision_contribution_test",

    "physics_informed_contribution_train",
    "physics_informed_contribution_val",
    "physics_informed_contribution_test",

    "total_force_train",
    "total_force_val",
    "total_force_test",

    "friction_obs_train",
    "friction_obs_val",
    "friction_obs_test",

    "lub_supervised_train",
    "lub_supervised_val",
    "lub_supervised_test",

    "lub_supervised_raw_train",
    "lub_supervised_raw_val",
    "lub_supervised_raw_test",

    "lub_supervised_raw_rmse_scaled_log_train",
    "lub_supervised_raw_rmse_scaled_log_val",
    "lub_supervised_raw_rmse_scaled_log_test",

    "lub_supervised_raw_rmse_log_unit_train",
    "lub_supervised_raw_rmse_log_unit_val",
    "lub_supervised_raw_rmse_log_unit_test",

    "lub_prior_train",
    "lub_prior_val",
    "lub_prior_test",

    "lub_prior_raw_train",
    "lub_prior_raw_val",
    "lub_prior_raw_test",

    "lub_prior_raw_rmse_standardized_log_train",
    "lub_prior_raw_rmse_standardized_log_val",
    "lub_prior_raw_rmse_standardized_log_test",

    "lub_prior_raw_rmse_log_unit_train",
    "lub_prior_raw_rmse_log_unit_val",
    "lub_prior_raw_rmse_log_unit_test",

    "lub_mae_train",
    "lub_mae_val",
    "lub_mae_test",

    "param_N0_train",
    "param_N0_val",
    "param_N0_test",

    "param_mu_train",
    "param_mu_val",
    "param_mu_test",

    "param_phi_train",
    "param_phi_val",
    "param_phi_test",

    "param_gamma_train",
    "param_gamma_val",
    "param_gamma_test",

    # -----------------------------------------------------
    # Raw and scaled parameter prediction metrics
    # -----------------------------------------------------
    "N0_mae_raw_train",
    "N0_mae_raw_val",
    "N0_mae_raw_test",

    "N0_mse_raw_train",
    "N0_mse_raw_val",
    "N0_mse_raw_test",

    "N0_rmse_raw_train",
    "N0_rmse_raw_val",
    "N0_rmse_raw_test",

    "N0_mse_scaled_train",
    "N0_mse_scaled_val",
    "N0_mse_scaled_test",

    "N0_rmse_scaled_train",
    "N0_rmse_scaled_val",
    "N0_rmse_scaled_test",

    "N0_mape_percent_train",
    "N0_mape_percent_val",
    "N0_mape_percent_test",

    "N0_r2_train",
    "N0_r2_val",
    "N0_r2_test",

    "N0_corr_train",
    "N0_corr_val",
    "N0_corr_test",

    "mu_mae_raw_train",
    "mu_mae_raw_val",
    "mu_mae_raw_test",

    "mu_mse_raw_train",
    "mu_mse_raw_val",
    "mu_mse_raw_test",

    "mu_rmse_raw_train",
    "mu_rmse_raw_val",
    "mu_rmse_raw_test",

    "mu_mse_scaled_train",
    "mu_mse_scaled_val",
    "mu_mse_scaled_test",

    "mu_rmse_scaled_train",
    "mu_rmse_scaled_val",
    "mu_rmse_scaled_test",

    "mu_mape_percent_train",
    "mu_mape_percent_val",
    "mu_mape_percent_test",

    "mu_r2_train",
    "mu_r2_val",
    "mu_r2_test",

    "mu_corr_train",
    "mu_corr_val",
    "mu_corr_test",

    "phi_deg_mae_raw_train",
    "phi_deg_mae_raw_val",
    "phi_deg_mae_raw_test",

    "phi_deg_mse_raw_train",
    "phi_deg_mse_raw_val",
    "phi_deg_mse_raw_test",

    "phi_deg_rmse_raw_train",
    "phi_deg_rmse_raw_val",
    "phi_deg_rmse_raw_test",

    "phi_deg_mse_scaled_train",
    "phi_deg_mse_scaled_val",
    "phi_deg_mse_scaled_test",

    "phi_deg_rmse_scaled_train",
    "phi_deg_rmse_scaled_val",
    "phi_deg_rmse_scaled_test",

    "phi_deg_mape_percent_train",
    "phi_deg_mape_percent_val",
    "phi_deg_mape_percent_test",

    "phi_deg_r2_train",
    "phi_deg_r2_val",
    "phi_deg_r2_test",

    "phi_deg_corr_train",
    "phi_deg_corr_val",
    "phi_deg_corr_test",

    "gamma_mae_raw_train",
    "gamma_mae_raw_val",
    "gamma_mae_raw_test",

    "gamma_mse_raw_train",
    "gamma_mse_raw_val",
    "gamma_mse_raw_test",

    "gamma_rmse_raw_train",
    "gamma_rmse_raw_val",
    "gamma_rmse_raw_test",

    "gamma_mse_scaled_train",
    "gamma_mse_scaled_val",
    "gamma_mse_scaled_test",

    "gamma_rmse_scaled_train",
    "gamma_rmse_scaled_val",
    "gamma_rmse_scaled_test",

    "gamma_mape_percent_train",
    "gamma_mape_percent_val",
    "gamma_mape_percent_test",

    "gamma_r2_train",
    "gamma_r2_val",
    "gamma_r2_test",

    "gamma_corr_train",
    "gamma_corr_val",
    "gamma_corr_test",

    # -----------------------------------------------------
    # Latent lubrication metrics
    # -----------------------------------------------------
    "Actual_F_lub_mae_raw_train",
    "Actual_F_lub_mae_raw_val",
    "Actual_F_lub_mae_raw_test",

    "Actual_F_lub_mse_raw_train",
    "Actual_F_lub_mse_raw_val",
    "Actual_F_lub_mse_raw_test",

    "Actual_F_lub_rmse_raw_train",
    "Actual_F_lub_rmse_raw_val",
    "Actual_F_lub_rmse_raw_test",

    "Actual_F_lub_mape_percent_train",
    "Actual_F_lub_mape_percent_val",
    "Actual_F_lub_mape_percent_test",

    "Actual_F_lub_r2_train",
    "Actual_F_lub_r2_val",
    "Actual_F_lub_r2_test",

    "Actual_F_lub_corr_train",
    "Actual_F_lub_corr_val",
    "Actual_F_lub_corr_test",

    "log_F_lub_mse_raw_train",
    "log_F_lub_mse_raw_val",
    "log_F_lub_mse_raw_test",

    "log_F_lub_rmse_raw_train",
    "log_F_lub_rmse_raw_val",
    "log_F_lub_rmse_raw_test",

    "log_F_lub_mse_scaled_train",
    "log_F_lub_mse_scaled_val",
    "log_F_lub_mse_scaled_test",

    "log_F_lub_rmse_scaled_train",
    "log_F_lub_rmse_scaled_val",
    "log_F_lub_rmse_scaled_test",

    # -----------------------------------------------------
    # Engineering metrics
    # -----------------------------------------------------
    "phi_mae_deg_train",
    "phi_mae_deg_val",
    "phi_mae_deg_test",

    "gamma_mae_train",
    "gamma_mae_val",
    "gamma_mae_test",

    "phi_within_tol_train",
    "phi_within_tol_val",
    "phi_within_tol_test",

    "gamma_within_tol_train",
    "gamma_within_tol_val",
    "gamma_within_tol_test",

    "face_target_train",
    "face_target_val",
    "face_target_test",

    "kfric_target_train",
    "kfric_target_val",
    "kfric_target_test",
]

compact_cols = [
    col for col in compact_cols
    if col in metrics_wide_df.columns
]

compact_summary_df = metrics_wide_df[compact_cols].copy()

# Add timing and parameter-count metadata to the direct comparison table.
training_timing_summary_df = collect_training_timing_summaries(
    stages=STAGES_TO_EVALUATE
)

if len(training_timing_summary_df) > 0:
    timing_merge_cols = [
        "trained_stage",
        "architecture",
        "model_version",
        "window_size",
        "batch_size",
        "max_window_size_for_comparison",
        "common_target_start_position",
        "use_parameter_supervision",
        "use_lubrication_supervision",
        "use_physics_informed_loss",
        "training_time_seconds",
        "training_time_minutes",
        "average_epoch_time_seconds",
        "average_train_epoch_time_seconds",
        "average_validation_epoch_time_seconds",
        "average_seconds_per_train_window",
        "average_train_windows_per_second",
        "time_to_best_checkpoint_seconds",
        "time_to_best_checkpoint_minutes",
        "epochs_completed",
        "best_epoch",
        "total_parameters",
        "trainable_parameters_at_stage_end",
        "num_train_windows",
        "num_validation_windows",
        "num_test_windows",
    ]
    timing_merge_cols = [
        col for col in timing_merge_cols
        if col in training_timing_summary_df.columns
    ]
    timing_merge_df = training_timing_summary_df[
        timing_merge_cols
    ].drop_duplicates(
        subset=[
            "trained_stage",
            "window_size",
            "batch_size",
        ]
    )
    compact_summary_df = compact_summary_df.merge(
        timing_merge_df,
        on=[
            "trained_stage",
            "window_size",
            "batch_size",
        ],
        how="left",
    )

inference_timing_cols = [
    col for col in [
        "inference_time_seconds_test",
        "inference_windows_test",
        "inference_ms_per_window_test",
        "inference_windows_per_second_test",
    ]
    if col in metrics_wide_df.columns
]

if len(inference_timing_cols) > 0:
    compact_summary_df = compact_summary_df.merge(
        metrics_wide_df[
            [
                "trained_stage",
                "window_size",
                "batch_size",
            ] + inference_timing_cols
        ].drop_duplicates(
            subset=[
                "trained_stage",
                "window_size",
                "batch_size",
            ]
        ),
        on=[
            "trained_stage",
            "window_size",
            "batch_size",
        ],
        how="left",
    )


history_all_df = collect_all_stage_histories(
    stages=STAGES_TO_EVALUATE
)

best_epoch_summary_df = summarize_best_epoch_from_history(
    history_all_df=history_all_df
)



if len(missing_stages) > 0:
    missing_stage_df = pd.DataFrame({
        "missing_stage": missing_stages,
        "expected_checkpoint_path": [
            get_stage_checkpoint_path(stage)
            for stage in missing_stages
        ],
    })
else:
    missing_stage_df = pd.DataFrame()



with pd.ExcelWriter(all_stage_excel_path, engine="openpyxl") as writer:
    compact_summary_df.to_excel(
        writer,
        sheet_name="compact_summary",
        index=False,
    )

    loss_summary_df.to_excel(
        writer,
        sheet_name="all_loss_summary",
        index=False,
    )

    prediction_summary_df.to_excel(
        writer,
        sheet_name="prediction_summary",
        index=False,
    )

    metrics_wide_df.to_excel(
        writer,
        sheet_name="metrics_wide",
        index=False,
    )

    metrics_long_df.to_excel(
        writer,
        sheet_name="metrics_long",
        index=False,
    )

    if len(best_epoch_summary_df) > 0:
        best_epoch_summary_df.to_excel(
            writer,
            sheet_name="best_epoch_summary",
            index=False,
        )

    if len(history_all_df) > 0:
        history_all_df.to_excel(
            writer,
            sheet_name="training_history_all",
            index=False,
        )

    if len(training_timing_summary_df) > 0:
        training_timing_summary_df.to_excel(
            writer,
            sheet_name="training_timing_summary",
            index=False,
        )

    if len(missing_stage_df) > 0:
        missing_stage_df.to_excel(
            writer,
            sheet_name="missing_stages",
            index=False,
        )


try:
    from openpyxl import load_workbook
    from openpyxl.styles import Font, Alignment
    from openpyxl.utils import get_column_letter

    wb = load_workbook(all_stage_excel_path)

    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center")

        for col_idx, column_cells in enumerate(ws.columns, start=1):
            max_length = 0
            col_letter = get_column_letter(col_idx)

            for cell in column_cells:
                try:
                    cell_value = str(cell.value)
                    max_length = max(max_length, len(cell_value))
                except Exception:
                    pass

            adjusted_width = min(max(max_length + 2, 12), 35)
            ws.column_dimensions[col_letter].width = adjusted_width

    wb.save(all_stage_excel_path)

except Exception as e:
    print("\nExcel formatting step was skipped because of this error:")
    print(e)


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)

print("\n" + "=" * 120)
print(f"{MODEL_VERSION.upper()} TRAINING / VALIDATION / TEST SUMMARY")
print("=" * 120)
print(f"Stages requested: {STAGES_TO_EVALUATE}")
print(f"Excel saved to:   {all_stage_excel_path}")

if len(missing_stages) > 0:
    print(f"Missing stages skipped: {missing_stages}")

print("=" * 120)

print("\nCompact all-stage comparison:")
print(compact_summary_df.round(6).to_string(index=False))
display(compact_summary_df.round(6))

print("\nAll loss summary:")
print(loss_summary_df.round(6).to_string(index=False))
display(loss_summary_df.round(6))

print("\nPrediction metric summary with MAPE, R2, correlation, raw MSE/RMSE, and scaled MSE/RMSE:")
print(prediction_summary_df.round(6).to_string(index=False))
display(prediction_summary_df.round(6))

if len(best_epoch_summary_df) > 0:
    best_display_cols = [
        "trained_stage",
        "epoch",
        "monitor_col",
        "best_monitor_value_from_history",

        "train_total",
        "val_total",
        "train_param",
        "val_param",

        "train_total_force",
        "val_total_force",
        "train_friction_obs",
        "val_friction_obs",

        "train_lub_supervised",
        "val_lub_supervised",
        "train_lub_supervised_raw",
        "val_lub_supervised_raw",
        "train_lub_supervised_raw_rmse_scaled_log",
        "val_lub_supervised_raw_rmse_scaled_log",
        "train_lub_supervised_raw_rmse_log_unit",
        "val_lub_supervised_raw_rmse_log_unit",

        "train_lub_prior",
        "val_lub_prior",
        "train_lub_prior_raw",
        "val_lub_prior_raw",
        "train_lub_prior_raw_rmse_standardized_log",
        "val_lub_prior_raw_rmse_standardized_log",
        "train_lub_prior_raw_rmse_log_unit",
        "val_lub_prior_raw_rmse_log_unit",

        "train_lub_mae",
        "val_lub_mae",

        "train_phi_mae_deg",
        "val_phi_mae_deg",
        "train_gamma_mae",
        "val_gamma_mae",
    ]

    best_display_cols = [
        col for col in best_display_cols
        if col in best_epoch_summary_df.columns
    ]

    print("\nBest epoch summary from NN training histories:")
    print(best_epoch_summary_df[best_display_cols].round(6).to_string(index=False))
    display(best_epoch_summary_df[best_display_cols].round(6))

print("\nDetailed wide-format metric table:")
display(metrics_wide_df.round(6))

print("\n" + "=" * 120)
print(f"15 COMPLETE | {MODEL_VERSION.upper()}")
print("=" * 120)





# =========================================================
# BLOCK 16 Test-only all-stage summary bar chart
# =========================================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


# ---------------------------------------------------------
# user settings
# ---------------------------------------------------------
SHOW_FIGURES_IN_CELL = True
SAVE_FIGURES_AS_PDF = True
SAVE_FIGURES_AS_PNG = True

stage_bar_figure_dir = os.path.join(
    save_dir,
    "all_stage_parameter_bar_charts"
)
os.makedirs(stage_bar_figure_dir, exist_ok=True)

test_summary_figure_pdf_path = os.path.join(
    stage_bar_figure_dir,
    "all_stage_test_metrics_four_parameters_2x2.pdf"
)

test_summary_figure_png_path = os.path.join(
    stage_bar_figure_dir,
    "all_stage_test_metrics_four_parameters_2x2.png"
)

test_summary_plot_data_path = os.path.join(
    stage_bar_figure_dir,
    "all_stage_test_metrics_four_parameters_2x2_plot_data.xlsx"
)


# ---------------------------------------------------------
# Bar style controls
# ---------------------------------------------------------
BAR_WIDTH = 0.34

COLOR_LEFT_FILL = "#C8C8C8"
COLOR_LEFT_EDGE = "#5A5A5A"

COLOR_RIGHT_FILL = "#E8A8AD"
COLOR_RIGHT_EDGE = "#800000"

COLOR_MAIN_FRAME = "#777777"

VALUE_LABEL_FONTSIZE = 7.0
AXIS_LABEL_FONTSIZE = 10.5

# Controls how much extra vertical space is left for vertical value labels.
LEFT_AXIS_TOP_PADDING_FACTOR = 1.60
RIGHT_AXIS_TOP_PADDING_FACTOR = 1.30

print("\n" + "=" * 100)
print("BLOCK 16 TEST-ONLY ALL-STAGE SUMMARY BAR CHART")
print("=" * 100)
print(f"Output folder: {stage_bar_figure_dir}")
print(f"PDF output:    {test_summary_figure_pdf_path}")
print(f"PNG output:    {test_summary_figure_png_path}")
print(f"Data output:   {test_summary_plot_data_path}")
print("=" * 100)



plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": AXIS_LABEL_FONTSIZE,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "mathtext.fontset": "dejavuserif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 1.1,
})



def load_compact_summary_df():
    if "compact_summary_df" in globals():
        print("Using existing compact_summary_df from memory.")
        return compact_summary_df.copy()

    candidate_paths = []

    candidate_var_names = [
        "all_stage_excel_path",
        "all_stage_summary_output_path",
        "training_validation_test_summary_excel_path",
        "summary_output_path",
        "final_excel_output_path",
    ]

    for var_name in candidate_var_names:
        if var_name in globals():
            path_value = globals()[var_name]
            if isinstance(path_value, str) and os.path.exists(path_value):
                candidate_paths.append(path_value)

    candidate_filenames = [
        "V4p1p10_all_stage_training_validation_test_summary.xlsx",
        "all_stage_training_validation_test_summary.xlsx",
        "version_4p1p10_all_stage_training_validation_test_summary.xlsx",
        "all_stage_summary.xlsx",
    ]

    for filename in candidate_filenames:
        candidate_paths.append(os.path.join(save_dir, filename))

    candidate_paths.extend(
        glob.glob(
            os.path.join(save_dir, "**", "*stage*summary*.xlsx"),
            recursive=True,
        )
    )

    candidate_paths.extend(
        glob.glob(
            os.path.join(save_dir, "**", "*training*validation*test*.xlsx"),
            recursive=True,
        )
    )

    seen = set()
    candidate_paths_unique = []

    for path in candidate_paths:
        if path not in seen and os.path.exists(path):
            candidate_paths_unique.append(path)
            seen.add(path)

    if len(candidate_paths_unique) == 0:
        raise FileNotFoundError(
            "Could not find compact_summary_df in memory or a saved summary Excel file.\n"
            "Please run Block 15 first, or verify the Excel summary path."
        )

    for excel_path in candidate_paths_unique:
        try:
            xls = pd.ExcelFile(excel_path)

            if "compact_summary" in xls.sheet_names:
                df = pd.read_excel(excel_path, sheet_name="compact_summary")
                print("Loaded compact_summary from:")
                print(excel_path)
                return df

            if "compact summary" in xls.sheet_names:
                df = pd.read_excel(excel_path, sheet_name="compact summary")
                print("Loaded compact summary from:")
                print(excel_path)
                return df

            df = pd.read_excel(excel_path, sheet_name=xls.sheet_names[0])
            print("Loaded first sheet from:")
            print(excel_path)
            return df

        except Exception as e:
            print(f"Skipped Excel candidate due to read error:\n{excel_path}\n{e}")

    raise RuntimeError(
        "Found possible Excel files, but none could be read successfully."
    )


stage_metric_df = load_compact_summary_df()



def standardize_stage_column(df):
    out = df.copy()

    if "trained_stage" in out.columns:
        stage_col = "trained_stage"
    elif "stage" in out.columns:
        out = out.rename(columns={"stage": "trained_stage"})
        stage_col = "trained_stage"
    elif "Stage" in out.columns:
        out = out.rename(columns={"Stage": "trained_stage"})
        stage_col = "trained_stage"
    else:
        raise ValueError(
            "Could not find a stage column. Expected 'trained_stage', 'stage', or 'Stage'."
        )

    out = out.sort_values(stage_col).reset_index(drop=True)

    return out


def ensure_accuracy_columns(df):
    out = df.copy()

    for quantity in ["N0", "mu"]:
        acc_col = f"{quantity}_accuracy_percent_test"
        mape_col = f"{quantity}_mape_percent_test"

        if acc_col not in out.columns:
            if mape_col not in out.columns:
                raise ValueError(
                    f"Missing both {acc_col} and {mape_col}. "
                    f"Cannot compute accuracy for {quantity}."
                )

            out[acc_col] = 100.0 - out[mape_col].astype(float)
            out[acc_col] = out[acc_col].clip(lower=0.0, upper=100.0)

    return out


def require_columns(df, required_cols):
    missing = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns are:\n"
            + "\n".join(df.columns.astype(str))
        )


stage_metric_df = standardize_stage_column(stage_metric_df)
stage_metric_df = ensure_accuracy_columns(stage_metric_df)

required_cols = [
    "trained_stage",

    "N0_mse_scaled_test",
    "N0_accuracy_percent_test",

    "mu_mse_scaled_test",
    "mu_accuracy_percent_test",

    "phi_deg_mae_raw_test",
    "phi_deg_mape_percent_test",

    "gamma_mae_raw_test",
    "gamma_mape_percent_test",
]

require_columns(stage_metric_df, required_cols)

print("\nStages found:")
print(stage_metric_df["trained_stage"].tolist())


plot_df = stage_metric_df[
    [
        "trained_stage",

        "N0_mse_scaled_test",
        "N0_accuracy_percent_test",

        "mu_mse_scaled_test",
        "mu_accuracy_percent_test",

        "phi_deg_mae_raw_test",
        "phi_deg_mape_percent_test",

        "gamma_mae_raw_test",
        "gamma_mape_percent_test",
    ]
].copy()



plot_df["N0_mse_test_display"] = plot_df["N0_mse_scaled_test"].astype(float)
plot_df["phi_deg_mae_raw_test_display"] = plot_df["phi_deg_mae_raw_test"].astype(float)
plot_df["phi_deg_mape_percent_test_display"] = plot_df["phi_deg_mape_percent_test"].astype(float)
plot_df["gamma_mae_raw_test_display"] = plot_df["gamma_mae_raw_test"].astype(float)
plot_df["gamma_mape_percent_test_display"] = plot_df["gamma_mape_percent_test"].astype(float)

# Save plot data.
plot_df.to_excel(test_summary_plot_data_path, index=False)



def format_mse(value):
    value = float(value)

    if abs(value) < 1e-3:
        return f"{value:.2e}"

    return f"{value:.3f}"

def format_n0_mse(value):
    value = float(value)
    return f"{value:.2e}"

def format_percent(value):
    return f"{float(value):.2f}%"


def truncate_float(value, decimals):
    value = float(value)
    factor = 10 ** decimals

    if value >= 0:
        return math.floor(value * factor) / factor
    else:
        return math.ceil(value * factor) / factor


def format_mae(value):
    return f"{truncate_float(value, 2):.2f}"

def format_mape(value):
    return f"{float(value):.2f}%"


def set_full_frame(ax, color=COLOR_MAIN_FRAME, linewidth=1.1):
    for spine_name in ["left", "bottom", "top", "right"]:
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color(color)
        ax.spines[spine_name].set_linewidth(linewidth)


def add_bar_value_labels(
    ax,
    bars,
    values,
    formatter,
    color="#333333",
    fontsize=VALUE_LABEL_FONTSIZE,
    y_offset_frac=0.010,
    rotation=90,
):
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    for bar, value in zip(bars, values):
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + y_offset_frac * y_range,
            formatter(value),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
            rotation=rotation,
            clip_on=True,
        )


def plot_dual_metric_stage_bars(
    ax,
    df,
    left_col,
    right_col,
    left_label,
    right_label,
    left_legend_label,
    right_legend_label,
    left_formatter,
    right_formatter,
    left_ylim=None,
    right_ylim=None,
):
    stage_values = df["trained_stage"].astype(int).values
    stage_labels = [str(s) for s in stage_values]

    x = np.arange(len(stage_values), dtype=float)

    left_values = df[left_col].astype(float).values
    right_values = df[right_col].astype(float).values

    ax2 = ax.twinx()

    left_bars = ax.bar(
        x - BAR_WIDTH / 2.0,
        left_values,
        width=BAR_WIDTH,
        color=COLOR_LEFT_FILL,
        edgecolor=COLOR_LEFT_EDGE,
        linewidth=1.0,
        alpha=0.86,
        label=left_legend_label,
        zorder=3,
    )

    right_bars = ax2.bar(
        x + BAR_WIDTH / 2.0,
        right_values,
        width=BAR_WIDTH,
        color=COLOR_RIGHT_FILL,
        edgecolor=COLOR_RIGHT_EDGE,
        linewidth=1.0,
        alpha=0.86,
        label=right_legend_label,
        zorder=3,
    )

    ax.set_xlabel("Pretraining stage")
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels)

    ax.set_ylabel(left_label)
    ax2.set_ylabel(right_label)

    if left_ylim is not None:
        ax.set_ylim(left_ylim)
    else:
        left_max = np.nanmax(left_values)
        if not np.isfinite(left_max) or left_max <= 0:
            left_max = 1.0
        ax.set_ylim(0.0, left_max * LEFT_AXIS_TOP_PADDING_FACTOR)

    if right_ylim is not None:
        ax2.set_ylim(right_ylim)
    else:
        right_max = np.nanmax(right_values)
        if not np.isfinite(right_max) or right_max <= 0:
            right_max = 1.0
        ax2.set_ylim(0.0, right_max * RIGHT_AXIS_TOP_PADDING_FACTOR)

    ax.grid(
        axis="y",
        linestyle="-",
        linewidth=0.6,
        alpha=0.18,
        color="#999999",
        zorder=0,
    )

    ax2.grid(False)

    ax.set_facecolor("white")
    ax2.set_facecolor("none")

    set_full_frame(ax)
    set_full_frame(ax2)

    ax2.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    add_bar_value_labels(
        ax=ax,
        bars=left_bars,
        values=left_values,
        formatter=left_formatter,
        color=COLOR_LEFT_EDGE,
        rotation=90,
    )

    add_bar_value_labels(
        ax=ax2,
        bars=right_bars,
        values=right_values,
        formatter=right_formatter,
        color=COLOR_RIGHT_EDGE,
        rotation=90,
    )

    handles_1, labels_1 = ax.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()

    ax.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#D9D9D9",
        handlelength=1.4,
        columnspacing=1.0,
        borderpad=0.25,
    )

    return ax, ax2



fig, axes = plt.subplots(
    2,
    2,
    figsize=(13.6, 9.2),
)

fig.patch.set_facecolor("white")


plot_dual_metric_stage_bars(
    ax=axes[0, 0],
    df=plot_df,
    left_col="N0_mse_test_display",
    right_col="N0_accuracy_percent_test",
    left_label="MSE of N0",
    right_label="Accuracy of N0 (%)",
    left_legend_label="MSE",
    right_legend_label="Accuracy",
    left_formatter=format_n0_mse,
    right_formatter=format_percent,
    left_ylim=None,
    right_ylim=(0.0, 125.0),
)



plot_dual_metric_stage_bars(
    ax=axes[0, 1],
    df=plot_df,
    left_col="mu_mse_scaled_test",
    right_col="mu_accuracy_percent_test",
    left_label="MSE of μ",
    right_label="Accuracy of μ (%)",
    left_legend_label="MSE",
    right_legend_label="Accuracy",
    left_formatter=format_mse,
    right_formatter=format_percent,
    left_ylim=None,
    right_ylim=(0.0, 125.0),
)



plot_dual_metric_stage_bars(
    ax=axes[1, 0],
    df=plot_df,
    left_col="phi_deg_mae_raw_test_display",
    right_col="phi_deg_mape_percent_test_display",
    left_label="MAE of φ (deg)",
    right_label="MAPE of φ (%)",
    left_legend_label="MAE",
    right_legend_label="MAPE",
    left_formatter=format_mae,
    right_formatter=format_mape,
    left_ylim=None,
    right_ylim=None,
)


plot_dual_metric_stage_bars(
    ax=axes[1, 1],
    df=plot_df,
    left_col="gamma_mae_raw_test_display",
    right_col="gamma_mape_percent_test_display",
    left_label="MAE of γ (kN/m³)",
    right_label="MAPE of γ (%)",
    left_legend_label="MAE",
    right_legend_label="MAPE",
    left_formatter=format_mae,
    right_formatter=format_mape,
    left_ylim=None,
    right_ylim=None,
)


subplot_labels = ["(a)", "(b)", "(c)", "(d)"]

for ax, label in zip(axes.flatten(), subplot_labels):
    ax.text(
        0.5,
        -0.20,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98], h_pad=3.1, w_pad=2.4)


if SAVE_FIGURES_AS_PDF:
    plt.savefig(
        test_summary_figure_pdf_path,
        format="pdf",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
        transparent=False,
    )

if SAVE_FIGURES_AS_PNG:
    plt.savefig(
        test_summary_figure_png_path,
        format="png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
        transparent=False,
    )

if SHOW_FIGURES_IN_CELL:
    plt.show()

plt.close()

print("\nSaved 2x2 test summary figure:")
if SAVE_FIGURES_AS_PDF:
    print(test_summary_figure_pdf_path)
if SAVE_FIGURES_AS_PNG:
    print(test_summary_figure_png_path)

print("\nSaved plot data:")
print(test_summary_plot_data_path)

print("\n" + "=" * 100)
print("BLOCK 16 TEST-ONLY ALL-STAGE SUMMARY BAR CHART COMPLETE")
print("=" * 100)




# =========================================================
# 17. zero-shot real-case test for all trained stages
# =========================================================

import os
import numpy as np
import pandas as pd
import torch
from IPython.display import display
import time

# ---------------------------------------------------------
# user settings
# ---------------------------------------------------------
REAL_START_JACKING_DISTANCE = 108.204

REAL_CASE_FILE = "Simulated_ESI_Lub_with_Lub.xlsx"
REAL_SHEET_NAME = 0

# Stage 1: N0
# Stage 2: lubrication
# Stage 3: mu
# Stage 4: phi
# Stage 5: gamma
# Stage 6: optional joint fine-tuning
REAL_STAGES_TO_TEST = [5]

REAL_BATCH_SIZE = 512

REAL_REF_N0 = 2.5
REAL_REF_MU = 0.58
REAL_BASE_PHI_DEG = 32.0
REAL_BASE_GAMMA = 17.28

REAL_BASE_USED_COLS = [
    "Base_phi_deg_used_for_inference",
    "Base_gamma_used_for_inference",
]

REAL_CONDITIONAL_PLACEHOLDER_MODE = "predicted"


REAL_PLACEHOLDER_MODE = "fixed"

real_all_stage_output_path = os.path.join(
    save_dir,
    "real_case_zero_shot_all_stages_V4p1p10_true_fric_lub_input.xlsx"
)



MAPE_EPS = 1e-8


def safe_ape_percent(pred, target, eps=MAPE_EPS):


    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    denominator = np.maximum(np.abs(target), eps)
    return np.abs((pred - target) / denominator) * 100.0


def safe_mape_percent(pred, target, eps=MAPE_EPS):


    return np.mean(
        safe_ape_percent(
            pred=pred,
            target=target,
            eps=eps,
        )
    )


def safe_r2_score(pred, target, eps=1e-12):


    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    ss_res = np.sum((pred - target) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)

    if ss_tot < eps:
        return np.nan

    return 1.0 - ss_res / ss_tot

def safe_rmse_from_mse(mse_value):

    return float(np.sqrt(float(mse_value)))
    

def get_real_stage_checkpoint_path(stage):
    return os.path.join(
        save_dir,
        f"best_multihead_cnn_stage_{stage}.pth"
    )


if REAL_CASE_FILE.endswith(".csv"):
    real_df = pd.read_csv(REAL_CASE_FILE)
else:
    real_df = pd.read_excel(REAL_CASE_FILE, sheet_name=REAL_SHEET_NAME)

real_df = real_df.copy()
real_df = real_df.loc[:, ~real_df.columns.duplicated()].copy()

print("\nRaw real-case columns:")
print(real_df.columns.tolist())



rename_map = {}

if "Face Pressure" in real_df.columns and "F_face" not in real_df.columns:
    rename_map["Face Pressure"] = "F_face"

if "Lubricated Forces" in real_df.columns and "Actual_F_lub" not in real_df.columns:
    rename_map["Lubricated Forces"] = "Actual_F_lub"

if "N_0" in real_df.columns and "N0" not in real_df.columns:
    rename_map["N_0"] = "N0"

real_df.rename(columns=rename_map, inplace=True)

print("\nApplied rename map:")
print(rename_map)

if "phi" in real_df.columns and "phi_deg" not in real_df.columns:
    if real_df["phi"].abs().median() < np.pi:
        real_df["phi_deg"] = np.degrees(real_df["phi"])
        print("\nConverted real-case phi from radians to phi_deg.")
    else:
        real_df["phi_deg"] = real_df["phi"]
        print("\nCopied real-case phi to phi_deg because it appears already in degrees.")


if "Jacking Distance" not in real_df.columns:
    raise ValueError(
        "The real-case file must contain 'Jacking Distance' "
        "before filtering the real-case segment."
    )

real_df = real_df[
    real_df["Jacking Distance"] >= REAL_START_JACKING_DISTANCE
].reset_index(drop=True)

print("\nFiltered real-case data:")
print(f"  Start jacking distance: {REAL_START_JACKING_DISTANCE}")
print(f"  Rows after filtering:   {len(real_df)}")

if len(real_df) == 0:
    raise ValueError(
        "No real-case rows remain after filtering by "
        f"Jacking Distance >= {REAL_START_JACKING_DISTANCE}."
    )

print("\nFiltered jacking-distance range:")
print(real_df["Jacking Distance"].describe())


if "Jacking Forces" not in real_df.columns:
    raise ValueError(
        "Observed 'Jacking Forces' is missing. Do not replace it with "
        "'Total Jacking Forces', because Total Jacking Forces is the "
        "no-lubrication total force in this real-case file."
    )

if "F_face" not in real_df.columns:
    raise ValueError(
        "The real-case file must contain 'Face Pressure' or 'F_face'. "
        "F_face is needed to compute True_F_fric_lub."
    )

real_df["True_F_fric_lub"] = (
    real_df["Jacking Forces"].values
    - real_df["F_face"].values
)

print("\nReal-case force construction:")
print("  Jacking Forces model input = observed lubricated Jacking Forces")
print("  True_F_fric_lub = observed Jacking Forces - F_face")
print("  Total Jacking Forces is kept only as diagnostic/no-lubrication reference.")


real_df[REAL_BASE_USED_COLS[0]] = REAL_BASE_PHI_DEG
real_df[REAL_BASE_USED_COLS[1]] = REAL_BASE_GAMMA

real_base_source = "fixed single-soil real-case prior/reference values"

print("\nVersion 4.1.10 real-case residual references:")
print(f"  Base source:       {real_base_source}")
print(f"  REF_N0:            {REAL_REF_N0:.6f}")
print(f"  REF_MU:            {REAL_REF_MU:.6f}")
print(f"  Base phi used:     {REAL_BASE_PHI_DEG:.6f}")
print(f"  Base gamma used:   {REAL_BASE_GAMMA:.6f}")
print(real_df[REAL_BASE_USED_COLS].describe())

if all(col in real_df.columns for col in ["Total Jacking Forces", "F_face"]):
    real_df["F_fric_unlub_from_total_minus_face"] = (
        real_df["Total Jacking Forces"].values
        - real_df["F_face"].values
    )

    print("\nNative friction diagnostic:")
    print("  F_fric_unlub_from_total_minus_face = Total Jacking Forces - F_face")

    if "Frictions" in real_df.columns:
        native_fric_check = (
            real_df["F_fric_unlub_from_total_minus_face"].values
            - real_df["Frictions"].values
        )

        print("  Compare against provided Frictions column")
        print(f"  mean difference:          {np.mean(native_fric_check):.6f}")
        print(f"  mean absolute difference: {np.mean(np.abs(native_fric_check)):.6f}")
        print(f"  max absolute difference:  {np.max(np.abs(native_fric_check)):.6f}")

if all(col in real_df.columns for col in ["Frictions", "Actual_F_lub"]):
    real_df["True_F_fric_lub_from_friction_minus_lub"] = (
        real_df["Frictions"].values
        - real_df["Actual_F_lub"].values
    )

    effective_fric_check = (
        real_df["True_F_fric_lub"].values
        - real_df["True_F_fric_lub_from_friction_minus_lub"].values
    )

    print("\nEffective friction diagnostic:")
    print("  Compare Jacking Forces - F_face against Frictions - Actual_F_lub")
    print(f"  mean difference:          {np.mean(effective_fric_check):.6f}")
    print(f"  mean absolute difference: {np.mean(np.abs(effective_fric_check)):.6f}")
    print(f"  max absolute difference:  {np.max(np.abs(effective_fric_check)):.6f}")

if all(col in real_df.columns for col in ["Total Jacking Forces", "Actual_F_lub"]):
    total_force_check = (
        real_df["Total Jacking Forces"].values
        - real_df["Actual_F_lub"].values
        - real_df["Jacking Forces"].values
    )

    print("\nTotal force diagnostic:")
    print("  Compare Total Jacking Forces - Actual_F_lub against observed Jacking Forces")
    print(f"  mean difference:          {np.mean(total_force_check):.6f}")
    print(f"  mean absolute difference: {np.mean(np.abs(total_force_check)):.6f}")
    print(f"  max absolute difference:  {np.max(np.abs(total_force_check)):.6f}")

missing_input_cols = [c for c in INPUT_COLS if c not in real_df.columns]

if missing_input_cols:
    raise ValueError(
        "The real-case file is missing required Version 4.1.10 model input columns:\n"
        f"{missing_input_cols}\n\n"
        "Version 4.1.10 requires observed Jacking Forces, F_face, and True_F_fric_lub. "
        "Actual_F_lub is not required as input.\n"
        f"Current columns are:\n{real_df.columns.tolist()}"
    )

real_df = real_df.dropna(subset=INPUT_COLS).reset_index(drop=True)

if len(real_df) == 0:
    raise ValueError(
        "No real-case rows remain after dropping rows with missing model inputs."
    )

print("\nVersion 4.1.10 model input columns:")
print(INPUT_COLS)

print("\nReal-case input summary:")
print(real_df[INPUT_COLS].describe())

print("\nNumber of real-case rows used:", len(real_df))


X_real_raw = real_df[INPUT_COLS].values.astype(np.float32)
X_real_scaled = scaler_X.transform(X_real_raw).astype(np.float32)


REAL_BASE_RAW = real_df[REAL_BASE_USED_COLS].values.astype(np.float32)



def make_padded_real_windows(X_scaled, window_size):
    n_rows, n_features = X_scaled.shape
    windows = []

    for i in range(n_rows):
        start = i - window_size + 1

        if start >= 0:
            window = X_scaled[start:i + 1]
        else:
            n_pad = -start
            pad_block = np.repeat(
                X_scaled[0:1],
                repeats=n_pad,
                axis=0,
            )
            window = np.vstack([pad_block, X_scaled[0:i + 1]])

        if window.shape[0] != window_size:
            raise RuntimeError(
                f"Window length mismatch at row {i}: "
                f"got {window.shape[0]}, expected {window_size}"
            )

        windows.append(window)

    return np.stack(windows, axis=0).astype(np.float32)


X_real_windows = make_padded_real_windows(
    X_scaled=X_real_scaled,
    window_size=window_size,
)

print("\nReal-case window tensor shape:")
print(X_real_windows.shape)


# ---------------------------------------------------------
# predict one real-case stage
# ---------------------------------------------------------

def synchronize_inference_timing():

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def predict_real_case_for_stage(stage):
    checkpoint_path = get_real_stage_checkpoint_path(stage)

    if not os.path.exists(checkpoint_path):
        print("\n" + "=" * 100)
        print(f"WARNING: NN Stage {stage} checkpoint was not found.")
        print("Expected checkpoint path:")
        print(checkpoint_path)
        print("This stage will be skipped.")
        print("=" * 100)
        return None, None

    print("\n" + "=" * 100)
    print(f"Running real-case zero-shot prediction for NN Stage {stage}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 100)

    # Load the checkpoint before timing.
    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
        )
    )

    model.to(device)
    model.eval()

    all_params_pred_scaled = []
    all_physics_pred_scaled = []

    all_log_lub_pred_scaled = []
    all_log_lub_pred_raw = []
    all_F_lub_pred_raw = []

  
    synchronize_inference_timing()
    inference_start_time = time.perf_counter()

    with torch.no_grad():
        for start in range(
            0,
            len(X_real_windows),
            REAL_BATCH_SIZE,
        ):
            end = start + REAL_BATCH_SIZE

            x_batch = torch.tensor(
                X_real_windows[start:end],
                dtype=torch.float32,
                device=device,
            )

            base_target_raw_batch = torch.tensor(
                REAL_BASE_RAW[start:end],
                dtype=torch.float32,
                device=device,
            )

            (
                params_pred_scaled_batch,
                physics_pred_scaled_batch,
                log_lub_pred_scaled_batch,
                log_lub_pred_raw_batch,
                F_lub_pred_raw_batch,
            ) = model(
                x_batch,
                detach_previous=(stage <= 5),
                base_target_raw=base_target_raw_batch,
                training_stage=stage,
                conditional_placeholder_mode=(
                    REAL_CONDITIONAL_PLACEHOLDER_MODE
                ),
            )

            all_params_pred_scaled.append(
                params_pred_scaled_batch.cpu().numpy()
            )

            all_physics_pred_scaled.append(
                physics_pred_scaled_batch.cpu().numpy()
            )

            all_log_lub_pred_scaled.append(
                log_lub_pred_scaled_batch.cpu().numpy()
            )

            all_log_lub_pred_raw.append(
                log_lub_pred_raw_batch.cpu().numpy()
            )

            all_F_lub_pred_raw.append(
                F_lub_pred_raw_batch.cpu().numpy()
            )

    params_pred_scaled = np.vstack(
        all_params_pred_scaled
    )

    physics_pred_scaled = np.vstack(
        all_physics_pred_scaled
    )

    log_lub_pred_scaled = np.vstack(
        all_log_lub_pred_scaled
    )

    log_lub_pred_raw = np.vstack(
        all_log_lub_pred_raw
    )

    F_lub_pred_raw = np.vstack(
        all_F_lub_pred_raw
    )

    params_pred_raw = (
        params_pred_scaled
        * scaler_Y.scale_.reshape(1, -1)
        + scaler_Y.mean_.reshape(1, -1)
    )

    physics_pred_raw = (
        physics_pred_scaled
        * scaler_P.scale_.reshape(1, -1)
        + scaler_P.mean_.reshape(1, -1)
    )


    synchronize_inference_timing()

    inference_time_seconds = (
        time.perf_counter()
        - inference_start_time
    )

    number_of_field_rows = len(real_df)
    number_of_prediction_windows = len(X_real_windows)

    print("\n" + "-" * 100)
    print(f"Stage {stage} real-case inference timing")
    print("-" * 100)
    print(
        f"Field observations processed: "
        f"{number_of_field_rows}"
    )
    print(
        f"Prediction windows processed: "
        f"{number_of_prediction_windows}"
    )
    print(
        f"Batch size:                    "
        f"{REAL_BATCH_SIZE}"
    )
    print(
        f"Total inference time:          "
        f"{inference_time_seconds:.6f} s"
    )
    print(
        f"Average time per window:       "
        f"{1000.0 * inference_time_seconds / max(number_of_prediction_windows, 1):.6f} ms"
    )
    print(
        f"Windows per second:            "
        f"{number_of_prediction_windows / max(inference_time_seconds, 1e-12):.2f}"
    )
    print(
        "Timing excludes checkpoint loading, "
        "metrics, plotting, and Excel writing."
    )
    print("-" * 100)

    prediction_pack = {
        "checkpoint_path": checkpoint_path,
        "params_pred_scaled": params_pred_scaled,
        "physics_pred_scaled": physics_pred_scaled,
        "log_lub_pred_scaled": log_lub_pred_scaled,
        "log_lub_pred_raw": log_lub_pred_raw,
        "F_lub_pred_raw": F_lub_pred_raw,
        "params_pred_raw": params_pred_raw,
        "physics_pred_raw": physics_pred_raw,
    }

    return prediction_pack, checkpoint_path



def build_real_prediction_dataframe(stage, prediction_pack):
    params_pred_raw = prediction_pack["params_pred_raw"]
    physics_pred_raw = prediction_pack["physics_pred_raw"]

    log_lub_pred_scaled = prediction_pack["log_lub_pred_scaled"]
    log_lub_pred_raw = prediction_pack["log_lub_pred_raw"]
    F_lub_pred_raw = prediction_pack["F_lub_pred_raw"]

    n_real = len(real_df)

    real_pred_df = pd.DataFrame(index=np.arange(n_real))

    real_pred_df["real_model_stage"] = np.full(
        n_real,
        stage,
        dtype=int,
    )

    real_pred_df["checkpoint_path"] = [
        prediction_pack["checkpoint_path"]
    ] * n_real

    for col in ["update_index", "n_obs_used"]:
        if col in real_df.columns:
            real_pred_df[col] = real_df[col].values

    for col in INPUT_COLS:
        real_pred_df[col] = real_df[col].values

    for col in REAL_BASE_USED_COLS:
        real_pred_df[col] = real_df[col].values

    for i, col in enumerate(PARAM_COLS):
        real_pred_df[f"{col}_pred"] = params_pred_raw[:, i]

    for i, col in enumerate(PHYSICS_COLS):
        real_pred_df[f"{col}_pred"] = physics_pred_raw[:, i]

    real_pred_df["log_F_lub_pred_scaled"] = log_lub_pred_scaled[:, 0]
    real_pred_df["log_F_lub_pred"] = log_lub_pred_raw[:, 0]
    real_pred_df["Actual_F_lub_pred"] = F_lub_pred_raw[:, 0]

    teacher_param_cols = ["N0", "mu", "phi_deg", "gamma"]

    for col in teacher_param_cols:
        if col in real_df.columns:
            real_pred_df[f"{col}_teacher"] = real_df[col].values

    sd_map = {
        "N0_sd": "N_0_sd",
        "mu_sd": "mu_sd",
        "phi_sd": "phi_sd",
        "gamma_sd": "gamma_sd",
    }

    for clean_name, source_col in sd_map.items():
        if source_col in real_df.columns:
            real_pred_df[clean_name] = real_df[source_col].values

    extra_real_cols = [
        "Frictions",
        "Friction",
        "F_face",
        "Jacking Forces",
        "Total Jacking Forces",
        "True_F_fric_lub",
        "F_fric_unlub_from_total_minus_face",
        "True_F_fric_lub_from_friction_minus_lub",
        "Actual_F_lub",
    ]

    for col in extra_real_cols:
        if col in real_df.columns and col not in real_pred_df.columns:
            real_pred_df[col] = real_df[col].values

    real_pred_df["True_F_fric_lub_from_Jacking_minus_Fface"] = (
        real_df["Jacking Forces"].values
        - real_df["F_face"].values
    )

    return real_pred_df



def add_force_reconstruction_diagnostics(stage, real_pred_df, prediction_pack):
    aux_available = all(col in real_df.columns for col in AUX_COLS)

    if not aux_available:
        return real_pred_df, {}

    params_pred_raw = prediction_pack["params_pred_raw"]
    F_lub_pred_raw = prediction_pack["F_lub_pred_raw"]

    A_real_raw = real_df[AUX_COLS].values.astype(np.float32)

    params_pred_tensor = torch.tensor(
        params_pred_raw,
        dtype=torch.float32,
        device=device,
    )

    if all(col in real_df.columns for col in ["N0", "mu", "phi_deg", "gamma"]):
        params_teacher_tensor = torch.tensor(
            real_df[["N0", "mu", "phi_deg", "gamma"]].values.astype(np.float32),
            dtype=torch.float32,
            device=device,
        )
    else:
        params_teacher_tensor = params_pred_tensor.clone()

    x_real_tensor = torch.tensor(
        X_real_raw,
        dtype=torch.float32,
        device=device,
    )

    a_real_tensor = torch.tensor(
        A_real_raw,
        dtype=torch.float32,
        device=device,
    )

    F_lub_pred_tensor = torch.tensor(
        F_lub_pred_raw,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        params_used_tensor = build_stage_physics_params(
            params_pred_raw=params_pred_tensor,
            y_target_raw=params_teacher_tensor,
            training_stage=stage,
            placeholder_mode=REAL_PLACEHOLDER_MODE,
            use_true_conditional_inputs=False,
        )

        force_terms = reconstruct_force_from_stage_params(
            params_used_for_physics=params_used_tensor,
            x_target_raw=x_real_tensor,
            a_target_raw=a_real_tensor,
            F_lub_pred_raw=F_lub_pred_tensor,
            placeholder_mode=REAL_PLACEHOLDER_MODE,
        )

    real_pred_df["F_face_formula_from_stage_params"] = (
        force_terms["F_face_formula"].cpu().numpy().ravel()
    )

    real_pred_df["K_fric_formula_from_stage_params"] = (
        force_terms["K_fric_formula"].cpu().numpy().ravel()
    )

    real_pred_df["F_fric_native_formula_from_stage_params"] = (
        force_terms["F_fric_native_formula"].cpu().numpy().ravel()
    )

    real_pred_df["F_fric_effective_formula_from_stage_params"] = (
        force_terms["F_fric_effective_formula"].cpu().numpy().ravel()
    )

    real_pred_df["Jacking_formula_from_stage_params"] = (
        force_terms["F_jacking_formula"].cpu().numpy().ravel()
    )

    real_pred_df["True_F_fric_lub_used_in_reconstruction"] = (
        force_terms["friction_obs"].cpu().numpy().ravel()
    )

    real_pred_df["True_F_fric_lub_input_minus_used"] = (
        real_pred_df["True_F_fric_lub"]
        - real_pred_df["True_F_fric_lub_used_in_reconstruction"]
    )

    jacking_error_raw = (
        real_pred_df["Jacking_formula_from_stage_params"].values
        - real_df["Jacking Forces"].values
    )

    friction_error_raw = (
        real_pred_df["F_fric_effective_formula_from_stage_params"].values
        - real_df["True_F_fric_lub"].values
    )

    try:
        global_jacking_std_value = float(global_jacking_std.detach().cpu().item())
    except Exception:
        global_jacking_std_value = float(scaler_X.scale_[INPUT_IDX["Jacking Forces"]])

    try:
        global_friction_obs_std_value = float(global_friction_obs_std.detach().cpu().item())
    except Exception:
        global_friction_obs_std_value = float(scaler_X.scale_[INPUT_IDX["True_F_fric_lub"]])

    if not np.isfinite(global_jacking_std_value) or global_jacking_std_value <= 0:
        global_jacking_std_value = 1.0

    if not np.isfinite(global_friction_obs_std_value) or global_friction_obs_std_value <= 0:
        global_friction_obs_std_value = 1.0

    jacking_error_scaled = jacking_error_raw / global_jacking_std_value
    friction_error_scaled = friction_error_raw / global_friction_obs_std_value

    real_pred_df["Jacking_formula_error_raw"] = jacking_error_raw
    real_pred_df["Jacking_formula_abs_error_raw"] = np.abs(jacking_error_raw)
    real_pred_df["Jacking_formula_sq_error_raw"] = jacking_error_raw ** 2

    real_pred_df["Jacking_formula_error_scaled"] = jacking_error_scaled
    real_pred_df["Jacking_formula_abs_error_scaled"] = np.abs(jacking_error_scaled)
    real_pred_df["Jacking_formula_sq_error_scaled"] = jacking_error_scaled ** 2

    real_pred_df["Jacking_formula_ape_percent"] = safe_ape_percent(
        pred=real_pred_df["Jacking_formula_from_stage_params"].values,
        target=real_df["Jacking Forces"].values,
    )

    real_pred_df["friction_effective_error_raw"] = friction_error_raw
    real_pred_df["friction_effective_abs_error_raw"] = np.abs(friction_error_raw)
    real_pred_df["friction_effective_sq_error_raw"] = friction_error_raw ** 2

    real_pred_df["friction_effective_error_scaled"] = friction_error_scaled
    real_pred_df["friction_effective_abs_error_scaled"] = np.abs(friction_error_scaled)
    real_pred_df["friction_effective_sq_error_scaled"] = friction_error_scaled ** 2

    real_pred_df["friction_effective_ape_percent"] = safe_ape_percent(
        pred=real_pred_df["F_fric_effective_formula_from_stage_params"].values,
        target=real_df["True_F_fric_lub"].values,
    )

    jacking_mse_raw = np.mean(jacking_error_raw ** 2)
    jacking_mse_scaled = np.mean(jacking_error_scaled ** 2)

    friction_mse_raw = np.mean(friction_error_raw ** 2)
    friction_mse_scaled = np.mean(friction_error_scaled ** 2)

    force_summary = {
        "jacking_formula_mae": np.mean(np.abs(jacking_error_raw)),
        "jacking_formula_mse_raw": jacking_mse_raw,
        "jacking_formula_rmse_raw": np.sqrt(jacking_mse_raw),
        "jacking_formula_bias": np.mean(jacking_error_raw),

        "jacking_formula_mse_scaled": jacking_mse_scaled,
        "jacking_formula_rmse_scaled": np.sqrt(jacking_mse_scaled),

        # Compatibility names
        "jacking_formula_mse": jacking_mse_raw,
        "jacking_formula_rmse": np.sqrt(jacking_mse_raw),
        "jacking_formula_norm_mse": jacking_mse_scaled,
        "jacking_formula_norm_rmse": np.sqrt(jacking_mse_scaled),

        "jacking_formula_mape_percent": safe_mape_percent(
            pred=real_pred_df["Jacking_formula_from_stage_params"].values,
            target=real_df["Jacking Forces"].values,
        ),
        "jacking_formula_r2": safe_r2_score(
            pred=real_pred_df["Jacking_formula_from_stage_params"].values,
            target=real_df["Jacking Forces"].values,
        ),

        "friction_effective_mae": np.mean(np.abs(friction_error_raw)),
        "friction_effective_mse_raw": friction_mse_raw,
        "friction_effective_rmse_raw": np.sqrt(friction_mse_raw),
        "friction_effective_bias": np.mean(friction_error_raw),

        "friction_effective_mse_scaled": friction_mse_scaled,
        "friction_effective_rmse_scaled": np.sqrt(friction_mse_scaled),

        # Compatibility names
        "friction_effective_mse": friction_mse_raw,
        "friction_effective_rmse": np.sqrt(friction_mse_raw),
        "friction_effective_norm_mse": friction_mse_scaled,
        "friction_effective_norm_rmse": np.sqrt(friction_mse_scaled),

        "friction_effective_mape_percent": safe_mape_percent(
            pred=real_pred_df["F_fric_effective_formula_from_stage_params"].values,
            target=real_df["True_F_fric_lub"].values,
        ),
        "friction_effective_r2": safe_r2_score(
            pred=real_pred_df["F_fric_effective_formula_from_stage_params"].values,
            target=real_df["True_F_fric_lub"].values,
        ),
    }

    return real_pred_df, force_summary

def add_parameter_target_metrics(stage, real_pred_df, prediction_pack):
    target_available = all(
        col in real_df.columns for col in ["N0", "mu", "phi_deg", "gamma"]
    )

    summary = {
        "target_params_available": target_available,
    }

    if not target_available:
        return real_pred_df, summary

    params_pred_raw = prediction_pack["params_pred_raw"]

    target_params_raw = real_df[
        ["N0", "mu", "phi_deg", "gamma"]
    ].values.astype(np.float32)

    param_error_raw = params_pred_raw - target_params_raw
    param_abs_error_raw = np.abs(param_error_raw)
    param_sq_error_raw = param_error_raw ** 2

    y_scale = scaler_Y.scale_.reshape(1, -1)
    y_scale = np.where(y_scale <= 0, 1.0, y_scale)

    param_error_scaled = param_error_raw / y_scale
    param_abs_error_scaled = np.abs(param_error_scaled)
    param_sq_error_scaled = param_error_scaled ** 2

    config_np = get_stage_loss_config(stage, device)
    param_weights_np = config_np["param_weights"].detach().cpu().numpy()

    denom = np.sum(param_weights_np)

    if denom > 0:
        param_loss_each_row = np.sum(
            param_sq_error_scaled * param_weights_np.reshape(1, -1),
            axis=1,
        ) / (denom + 1e-8)

        real_case_param_loss = np.mean(param_loss_each_row)
    else:
        param_loss_each_row = np.zeros(len(real_df))
        real_case_param_loss = 0.0

    new_cols = {}

    for i, col in enumerate(PARAM_COLS):
        pred_i = params_pred_raw[:, i]
        target_i = target_params_raw[:, i]

        new_cols[f"{col}_target"] = target_i

        new_cols[f"{col}_target_error_raw"] = param_error_raw[:, i]
        new_cols[f"{col}_target_abs_error_raw"] = param_abs_error_raw[:, i]
        new_cols[f"{col}_target_sq_error_raw"] = param_sq_error_raw[:, i]

        new_cols[f"{col}_target_error_scaled"] = param_error_scaled[:, i]
        new_cols[f"{col}_target_abs_error_scaled"] = param_abs_error_scaled[:, i]
        new_cols[f"{col}_target_sq_error_scaled"] = param_sq_error_scaled[:, i]
        new_cols[f"{col}_target_mse_scaled"] = param_sq_error_scaled[:, i]


        new_cols[f"{col}_error_raw"] = param_error_raw[:, i]
        new_cols[f"{col}_abs_error_raw"] = param_abs_error_raw[:, i]
        new_cols[f"{col}_sq_error_raw"] = param_sq_error_raw[:, i]

        new_cols[f"{col}_error_scaled"] = param_error_scaled[:, i]
        new_cols[f"{col}_abs_error_scaled"] = param_abs_error_scaled[:, i]
        new_cols[f"{col}_sq_error_scaled"] = param_sq_error_scaled[:, i]
        new_cols[f"{col}_mse_scaled"] = param_sq_error_scaled[:, i]

        new_cols[f"{col}_target_ape_percent"] = safe_ape_percent(
            pred=pred_i,
            target=target_i,
        )

    new_cols["real_case_param_loss_row"] = param_loss_each_row

    phi_abs_error = param_abs_error_raw[:, 2]
    gamma_abs_error = param_abs_error_raw[:, 3]

    new_cols["phi_abs_error_deg"] = phi_abs_error
    new_cols["gamma_abs_error"] = gamma_abs_error

    phi_within_tol = (phi_abs_error <= PHI_TOL_DEG).astype(int)
    gamma_within_tol = (gamma_abs_error <= GAMMA_TOL).astype(int)

    new_cols["phi_within_tol"] = phi_within_tol
    new_cols["gamma_within_tol"] = gamma_within_tol

    phi_tol_values = {}
    gamma_tol_values = {}

    for tol in PHI_TOL_LIST:
        key = tol_key("phi", tol)
        values = (phi_abs_error <= tol).astype(int)
        new_cols[key] = values
        phi_tol_values[key] = values

    for tol in GAMMA_TOL_LIST:
        key = tol_key("gamma", tol)
        values = (gamma_abs_error <= tol).astype(int)
        new_cols[key] = values
        gamma_tol_values[key] = values

    mu_train_mean = scaler_Y.mean_[1]
    mu_train_std = scaler_Y.scale_[1]

    if not np.isfinite(mu_train_std) or mu_train_std <= 0:
        mu_train_std = 1.0

    mu_pred_minus_train_mean = params_pred_raw[:, 1] - mu_train_mean
    mu_target_minus_train_mean = target_params_raw[:, 1] - mu_train_mean

    mu_pred_z_from_train_mean = mu_pred_minus_train_mean / mu_train_std
    mu_target_z_from_train_mean = mu_target_minus_train_mean / mu_train_std

    new_cols["mu_pred_minus_train_mean"] = mu_pred_minus_train_mean
    new_cols["mu_target_minus_train_mean"] = mu_target_minus_train_mean

    new_cols["mu_pred_z_from_train_mean"] = mu_pred_z_from_train_mean
    new_cols["mu_target_z_from_train_mean"] = mu_target_z_from_train_mean

    cols_to_replace = [
        col for col in new_cols.keys()
        if col in real_pred_df.columns
    ]

    if len(cols_to_replace) > 0:
        real_pred_df = real_pred_df.drop(columns=cols_to_replace).copy()

    # Add all new columns at once.
    real_pred_df = pd.concat(
        [
            real_pred_df,
            pd.DataFrame(new_cols, index=real_pred_df.index),
        ],
        axis=1,
    ).copy()

    real_pred_df = real_pred_df.loc[
        :,
        ~real_pred_df.columns.duplicated(keep="last"),
    ].copy()

    # Raw MSE/RMSE
    N0_mse_raw = param_sq_error_raw[:, 0].mean()
    mu_mse_raw = param_sq_error_raw[:, 1].mean()
    phi_mse_raw = param_sq_error_raw[:, 2].mean()
    gamma_mse_raw = param_sq_error_raw[:, 3].mean()

    # Scaled MSE/RMSE
    N0_mse_scaled = param_sq_error_scaled[:, 0].mean()
    mu_mse_scaled = param_sq_error_scaled[:, 1].mean()
    phi_mse_scaled = param_sq_error_scaled[:, 2].mean()
    gamma_mse_scaled = param_sq_error_scaled[:, 3].mean()

    summary.update({
        "real_case_param_loss": real_case_param_loss,

        "N0_mse_raw": N0_mse_raw,
        "N0_rmse_raw": np.sqrt(N0_mse_raw),

        "mu_mse_raw": mu_mse_raw,
        "mu_rmse_raw": np.sqrt(mu_mse_raw),

        "phi_mse_raw": phi_mse_raw,
        "phi_rmse_raw": np.sqrt(phi_mse_raw),

        "gamma_mse_raw": gamma_mse_raw,
        "gamma_rmse_raw": np.sqrt(gamma_mse_raw),

        # Compatibility names from previous block.
        "phi_mse_deg": phi_mse_raw,
        "phi_rmse_deg": np.sqrt(phi_mse_raw),
        "gamma_mse": gamma_mse_raw,
        "gamma_rmse": np.sqrt(gamma_mse_raw),

        "N0_mse_scaled": N0_mse_scaled,
        "N0_rmse_scaled": np.sqrt(N0_mse_scaled),

        "mu_mse_scaled": mu_mse_scaled,
        "mu_rmse_scaled": np.sqrt(mu_mse_scaled),

        "phi_mse_scaled": phi_mse_scaled,
        "phi_rmse_scaled": np.sqrt(phi_mse_scaled),

        "gamma_mse_scaled": gamma_mse_scaled,
        "gamma_rmse_scaled": np.sqrt(gamma_mse_scaled),

        "N0_mae_raw": param_abs_error_raw[:, 0].mean(),
        "mu_mae_raw": param_abs_error_raw[:, 1].mean(),
        "phi_mae_deg": param_abs_error_raw[:, 2].mean(),
        "gamma_mae": param_abs_error_raw[:, 3].mean(),

        "N0_mape_percent": safe_mape_percent(params_pred_raw[:, 0], target_params_raw[:, 0]),
        "mu_mape_percent": safe_mape_percent(params_pred_raw[:, 1], target_params_raw[:, 1]),
        "phi_mape_percent": safe_mape_percent(params_pred_raw[:, 2], target_params_raw[:, 2]),
        "gamma_mape_percent": safe_mape_percent(params_pred_raw[:, 3], target_params_raw[:, 3]),

        "N0_r2": safe_r2_score(params_pred_raw[:, 0], target_params_raw[:, 0]),
        "mu_r2": safe_r2_score(params_pred_raw[:, 1], target_params_raw[:, 1]),
        "phi_r2": safe_r2_score(params_pred_raw[:, 2], target_params_raw[:, 2]),
        "gamma_r2": safe_r2_score(params_pred_raw[:, 3], target_params_raw[:, 3]),

        "phi_within_tol": phi_within_tol.mean(),
        "gamma_within_tol": gamma_within_tol.mean(),

        "mu_train_mean": mu_train_mean,
        "mu_train_std": mu_train_std,
        "mu_pred_mean": params_pred_raw[:, 1].mean(),
        "mu_target_mean": target_params_raw[:, 1].mean(),
        "mu_pred_std": params_pred_raw[:, 1].std(),
        "mu_target_std": target_params_raw[:, 1].std(),
        "mu_pred_abs_dist_from_train_mean": np.abs(mu_pred_minus_train_mean).mean(),
        "mu_target_abs_dist_from_train_mean": np.abs(mu_target_minus_train_mean).mean(),
    })

    for key, values in phi_tol_values.items():
        summary[key] = values.mean()

    for key, values in gamma_tol_values.items():
        summary[key] = values.mean()

    return real_pred_df, summary

def add_lubrication_metrics(stage, real_pred_df, prediction_pack):
    log_lub_pred_scaled = prediction_pack["log_lub_pred_scaled"]
    log_lub_pred_raw = prediction_pack["log_lub_pred_raw"]
    F_lub_pred_raw = prediction_pack["F_lub_pred_raw"]

    config_np = get_stage_loss_config(stage, device)

    lub_supervised_weight_stage = float(config_np["lub_supervised_weight"])
    lub_prior_weight_stage = float(config_np["lub_prior_weight"])

    try:
        prior_log_mu = float(MU_LOG_FLUB_PRIOR)
    except Exception:
        prior_log_mu = float(mu_log_flub_prior_tensor.detach().cpu().item())

    try:
        prior_log_sigma = float(SIGMA_LOG_FLUB_PRIOR)
    except Exception:
        prior_log_sigma = float(sigma_log_flub_prior_tensor.detach().cpu().item())

    if not np.isfinite(prior_log_sigma) or prior_log_sigma <= 0:
        prior_log_sigma = 1.0

    prior_standardized_error = (
        log_lub_pred_raw - prior_log_mu
    ) / (prior_log_sigma + 1e-12)

    prior_sq_error = prior_standardized_error ** 2

    lub_prior_raw_real = prior_sq_error.mean()
    lub_prior_raw_rmse_standardized_log = np.sqrt(lub_prior_raw_real)
    lub_prior_raw_rmse_log_unit = (
        lub_prior_raw_rmse_standardized_log * prior_log_sigma
    )

    lub_prior_real = lub_prior_weight_stage * lub_prior_raw_real
    total_contribution_lub_prior_real = lambda_lub_prior * lub_prior_real

    new_cols = {}

    new_cols["log_F_lub_prior_standardized_error"] = (
        prior_standardized_error[:, 0]
    )

    new_cols["log_F_lub_prior_standardized_sq_error"] = (
        prior_sq_error[:, 0]
    )

    summary = {
        "lub_target_available": "Actual_F_lub" in real_df.columns,

        "lub_prior_weight_stage": lub_prior_weight_stage,
        "lub_prior_raw_real": lub_prior_raw_real,
        "lub_prior_real": lub_prior_real,
        "total_contribution_lub_prior_real": total_contribution_lub_prior_real,
        "lub_prior_raw_rmse_standardized_log": lub_prior_raw_rmse_standardized_log,
        "lub_prior_raw_rmse_log_unit": lub_prior_raw_rmse_log_unit,

        "F_lub_pred_mean": F_lub_pred_raw.mean(),
        "F_lub_pred_std": F_lub_pred_raw.std(),
        "log_F_lub_pred_mean": log_lub_pred_raw.mean(),
        "log_F_lub_pred_std": log_lub_pred_raw.std(),
    }

    if "Actual_F_lub" not in real_df.columns:
        summary.update({
            "lub_supervised_weight_stage": lub_supervised_weight_stage,
            "lub_supervised_raw_real": np.nan,
            "lub_supervised_real": np.nan,
            "total_contribution_lub_supervised_real": np.nan,
        })

        cols_to_replace = [
            col for col in new_cols.keys()
            if col in real_pred_df.columns
        ]

        if len(cols_to_replace) > 0:
            real_pred_df = real_pred_df.drop(columns=cols_to_replace).copy()

        real_pred_df = pd.concat(
            [
                real_pred_df,
                pd.DataFrame(new_cols, index=real_pred_df.index),
            ],
            axis=1,
        ).copy()

        real_pred_df = real_pred_df.loc[
            :,
            ~real_pred_df.columns.duplicated(keep="last"),
        ].copy()

        return real_pred_df, summary

    F_lub_target_raw = real_df["Actual_F_lub"].values.astype(np.float32).reshape(-1, 1)

    F_lub_error_raw = F_lub_pred_raw - F_lub_target_raw
    F_lub_abs_error_raw = np.abs(F_lub_error_raw)
    F_lub_sq_error_raw = F_lub_error_raw ** 2

    F_lub_mae_raw = F_lub_abs_error_raw.mean()
    F_lub_mse_raw = F_lub_sq_error_raw.mean()
    F_lub_rmse_raw = np.sqrt(F_lub_mse_raw)
    F_lub_bias_raw = F_lub_error_raw.mean()
    F_lub_mape_percent = safe_mape_percent(F_lub_pred_raw[:, 0], F_lub_target_raw[:, 0])
    F_lub_r2 = safe_r2_score(F_lub_pred_raw[:, 0], F_lub_target_raw[:, 0])

    new_cols["Actual_F_lub_target"] = F_lub_target_raw[:, 0]
    new_cols["Actual_F_lub_error_raw"] = F_lub_error_raw[:, 0]
    new_cols["Actual_F_lub_abs_error_raw"] = F_lub_abs_error_raw[:, 0]
    new_cols["Actual_F_lub_sq_error_raw"] = F_lub_sq_error_raw[:, 0]

    new_cols["Actual_F_lub_ape_percent"] = safe_ape_percent(
        pred=F_lub_pred_raw[:, 0],
        target=F_lub_target_raw[:, 0],
    )

    log_lub_target_raw = np.log(np.clip(F_lub_target_raw, LUB_EPS, None))
    log_lub_error_raw = log_lub_pred_raw - log_lub_target_raw
    log_lub_abs_error_raw = np.abs(log_lub_error_raw)
    log_lub_sq_error_raw = log_lub_error_raw ** 2

    log_lub_mae_raw = log_lub_abs_error_raw.mean()
    log_lub_mse_raw = log_lub_sq_error_raw.mean()
    log_lub_rmse_raw = np.sqrt(log_lub_mse_raw)
    log_lub_r2_raw = safe_r2_score(log_lub_pred_raw[:, 0], log_lub_target_raw[:, 0])

    new_cols["log_F_lub_target"] = log_lub_target_raw[:, 0]
    new_cols["log_F_lub_error"] = log_lub_error_raw[:, 0]
    new_cols["log_F_lub_abs_error"] = log_lub_abs_error_raw[:, 0]
    new_cols["log_F_lub_sq_error"] = log_lub_sq_error_raw[:, 0]

    log_lub_target_scaled = scaler_L.transform(log_lub_target_raw).astype(np.float32)

    log_lub_scaled_error = log_lub_pred_scaled - log_lub_target_scaled
    log_lub_scaled_abs_error = np.abs(log_lub_scaled_error)
    log_lub_scaled_sq_error = log_lub_scaled_error ** 2

    log_lub_scaled_mae = log_lub_scaled_abs_error.mean()
    log_lub_scaled_mse = log_lub_scaled_sq_error.mean()
    log_lub_scaled_rmse = np.sqrt(log_lub_scaled_mse)
    log_lub_scaled_r2 = safe_r2_score(
        log_lub_pred_scaled[:, 0],
        log_lub_target_scaled[:, 0],
    )

    new_cols["log_F_lub_target_scaled"] = log_lub_target_scaled[:, 0]
    new_cols["log_F_lub_scaled_error"] = log_lub_scaled_error[:, 0]
    new_cols["log_F_lub_scaled_abs_error"] = log_lub_scaled_abs_error[:, 0]
    new_cols["log_F_lub_scaled_sq_error"] = log_lub_scaled_sq_error[:, 0]

    lub_supervised_raw_real = log_lub_scaled_mse
    lub_supervised_real = lub_supervised_weight_stage * lub_supervised_raw_real
    total_contribution_lub_supervised_real = (
        lambda_lub_supervised * lub_supervised_real
    )

    summary.update({
        "lub_supervised_weight_stage": lub_supervised_weight_stage,

        "F_lub_mae_raw": F_lub_mae_raw,
        "F_lub_mse_raw": F_lub_mse_raw,
        "F_lub_rmse_raw": F_lub_rmse_raw,
        "F_lub_bias_raw": F_lub_bias_raw,
        "F_lub_mape_percent": F_lub_mape_percent,
        "F_lub_r2": F_lub_r2,
        "F_lub_target_mean": F_lub_target_raw.mean(),
        "F_lub_target_std": F_lub_target_raw.std(),

        "log_F_lub_mae_raw": log_lub_mae_raw,
        "log_F_lub_mse_raw": log_lub_mse_raw,
        "log_F_lub_rmse_raw": log_lub_rmse_raw,
        "log_F_lub_r2_raw": log_lub_r2_raw,

        "log_F_lub_scaled_mae": log_lub_scaled_mae,
        "log_F_lub_scaled_mse": log_lub_scaled_mse,
        "log_F_lub_scaled_rmse": log_lub_scaled_rmse,
        "log_F_lub_scaled_r2": log_lub_scaled_r2,

        "lub_supervised_raw_real": lub_supervised_raw_real,
        "lub_supervised_real": lub_supervised_real,
        "total_contribution_lub_supervised_real": (
            total_contribution_lub_supervised_real
        ),

        "lub_supervised_raw_rmse_scaled_log": log_lub_scaled_rmse,
        "lub_supervised_raw_rmse_log_unit": (
            log_lub_scaled_rmse * float(scaler_L.scale_[0])
        ),
    })

    cols_to_replace = [
        col for col in new_cols.keys()
        if col in real_pred_df.columns
    ]

    if len(cols_to_replace) > 0:
        real_pred_df = real_pred_df.drop(columns=cols_to_replace).copy()

    real_pred_df = pd.concat(
        [
            real_pred_df,
            pd.DataFrame(new_cols, index=real_pred_df.index),
        ],
        axis=1,
    ).copy()

    real_pred_df = real_pred_df.loc[
        :,
        ~real_pred_df.columns.duplicated(keep="last"),
    ].copy()

    return real_pred_df, summary


all_prediction_dfs = []
stage_summary_rows = []
missing_stages = []

for stage in REAL_STAGES_TO_TEST:
    checkpoint_path = get_real_stage_checkpoint_path(stage)

    if not os.path.exists(checkpoint_path):
        missing_stages.append(stage)
        print("\n" + "=" * 100)
        print(f"WARNING: NN Stage {stage} checkpoint missing. Skipping.")
        print(checkpoint_path)
        print("=" * 100)
        continue

    prediction_pack, _ = predict_real_case_for_stage(stage)

    if prediction_pack is None:
        missing_stages.append(stage)
        continue

    real_pred_stage_df = build_real_prediction_dataframe(
        stage=stage,
        prediction_pack=prediction_pack,
    )

    real_pred_stage_df, force_summary = add_force_reconstruction_diagnostics(
        stage=stage,
        real_pred_df=real_pred_stage_df,
        prediction_pack=prediction_pack,
    )

    real_pred_stage_df, param_summary = add_parameter_target_metrics(
        stage=stage,
        real_pred_df=real_pred_stage_df,
        prediction_pack=prediction_pack,
    )

    real_pred_stage_df, lub_summary = add_lubrication_metrics(
        stage=stage,
        real_pred_df=real_pred_stage_df,
        prediction_pack=prediction_pack,
    )

    summary_row = {
        "real_model_stage": stage,
        "checkpoint_path": prediction_pack["checkpoint_path"],
        "n_real_rows": len(real_pred_stage_df),
    }

    summary_row.update(param_summary)
    summary_row.update(lub_summary)
    summary_row.update(force_summary)

    stage_summary_rows.append(summary_row)
    all_prediction_dfs.append(real_pred_stage_df)

    print("\n" + "-" * 100)
    print(f"NN Stage {stage} real-case summary")
    print("-" * 100)

    print(f"Rows:                         {summary_row['n_real_rows']}")

    if summary_row.get("target_params_available", False):
        print(f"Real-case param loss:          {summary_row['real_case_param_loss']:.6f}")
        print(f"N0 raw MSE:                    {summary_row['N0_mse_raw']:.6f}")
        print(f"N0 raw RMSE:                   {summary_row['N0_rmse_raw']:.6f}")
        print(f"N0 scaled MSE:                 {summary_row['N0_mse_scaled']:.6f}")
        print(f"N0 scaled RMSE:                {summary_row['N0_rmse_scaled']:.6f}")
        print(f"N0 MAPE:                       {summary_row['N0_mape_percent']:.6f}%")
        print(f"N0 R2:                         {summary_row['N0_r2']:.6f}")

        print(f"mu raw MSE:                    {summary_row['mu_mse_raw']:.6f}")
        print(f"mu raw RMSE:                   {summary_row['mu_rmse_raw']:.6f}")
        print(f"mu scaled MSE:                 {summary_row['mu_mse_scaled']:.6f}")
        print(f"mu scaled RMSE:                {summary_row['mu_rmse_scaled']:.6f}")
        print(f"mu MAPE:                       {summary_row['mu_mape_percent']:.6f}%")
        print(f"mu R2:                         {summary_row['mu_r2']:.6f}")

        print(f"phi raw MSE:                   {summary_row['phi_mse_raw']:.6f}")
        print(f"phi raw RMSE:                  {summary_row['phi_rmse_raw']:.6f}")
        print(f"phi scaled MSE:                {summary_row['phi_mse_scaled']:.6f}")
        print(f"phi scaled RMSE:               {summary_row['phi_rmse_scaled']:.6f}")
        print(f"phi MAPE:                      {summary_row['phi_mape_percent']:.6f}%")
        print(f"phi R2:                        {summary_row['phi_r2']:.6f}")

        print(f"gamma raw MSE:                 {summary_row['gamma_mse_raw']:.6f}")
        print(f"gamma raw RMSE:                {summary_row['gamma_rmse_raw']:.6f}")
        print(f"gamma scaled MSE:              {summary_row['gamma_mse_scaled']:.6f}")
        print(f"gamma scaled RMSE:             {summary_row['gamma_rmse_scaled']:.6f}")
        print(f"gamma MAPE:                    {summary_row['gamma_mape_percent']:.6f}%")
        print(f"gamma R2:                      {summary_row['gamma_r2']:.6f}")

        print(f"Phi within tol:                {summary_row['phi_within_tol']:.6f}")
        print(f"Gamma within tol:              {summary_row['gamma_within_tol']:.6f}")
        
    if summary_row.get("lub_target_available", False):
        print(f"F_lub MAE raw:                 {summary_row['F_lub_mae_raw']:.6f}")
        print(f"F_lub RMSE raw:                {summary_row['F_lub_rmse_raw']:.6f}")
        print(f"F_lub MAPE:                    {summary_row['F_lub_mape_percent']:.6f}%")
        print(f"F_lub R2:                      {summary_row['F_lub_r2']:.6f}")
        print(f"scaled log_F_lub MSE:          {summary_row['log_F_lub_scaled_mse']:.6f}")
        print(f"scaled log_F_lub RMSE:         {summary_row['log_F_lub_scaled_rmse']:.6f}")
        print(f"scaled log_F_lub R2:           {summary_row['log_F_lub_scaled_r2']:.6f}")

    print(f"lub_prior_raw_real:            {summary_row['lub_prior_raw_real']:.6f}")
    print(f"lub_prior_rmse_standardized:   {summary_row['lub_prior_raw_rmse_standardized_log']:.6f}")

    if "jacking_formula_rmse_raw" in summary_row:
        print("\nForce reconstruction raw and scaled errors:")
        print(f"Jacking raw MSE:               {summary_row['jacking_formula_mse_raw']:.6f}")
        print(f"Jacking raw RMSE:              {summary_row['jacking_formula_rmse_raw']:.6f}")
        print(f"Jacking scaled MSE:            {summary_row['jacking_formula_mse_scaled']:.6f}")
        print(f"Jacking scaled RMSE:           {summary_row['jacking_formula_rmse_scaled']:.6f}")
        print(f"Jacking MAPE:                  {summary_row['jacking_formula_mape_percent']:.6f}%")
        print(f"Jacking R2:                    {summary_row['jacking_formula_r2']:.6f}")

        print(f"Friction raw MSE:              {summary_row['friction_effective_mse_raw']:.6f}")
        print(f"Friction raw RMSE:             {summary_row['friction_effective_rmse_raw']:.6f}")
        print(f"Friction scaled MSE:           {summary_row['friction_effective_mse_scaled']:.6f}")
        print(f"Friction scaled RMSE:          {summary_row['friction_effective_rmse_scaled']:.6f}")
        print(f"Friction MAPE:                 {summary_row['friction_effective_mape_percent']:.6f}%")
        print(f"Friction R2:                   {summary_row['friction_effective_r2']:.6f}")

    print("-" * 100)


if len(all_prediction_dfs) == 0:
    raise RuntimeError(
        "No real-case stages were evaluated. Please check the selected stages "
        "and checkpoint files."
    )

all_stage_predictions_df = pd.concat(
    all_prediction_dfs,
    ignore_index=True,
)

# Safety check: real_model_stage should not contain NaN.
if "real_model_stage" in all_stage_predictions_df.columns:
    if all_stage_predictions_df["real_model_stage"].isna().any():
        raise ValueError(
            "real_model_stage still contains NaN. "
            "Please check build_real_prediction_dataframe()."
        )

    all_stage_predictions_df["real_model_stage"] = (
        all_stage_predictions_df["real_model_stage"].astype(int)
    )

stage_summary_df = pd.DataFrame(stage_summary_rows)



compact_cols = [
    "real_model_stage",
    "n_real_rows",

    "real_case_param_loss",

    # -----------------------------------------------------
    # Parameter raw MSE/RMSE
    # -----------------------------------------------------
    "N0_mse_raw",
    "N0_rmse_raw",
    "mu_mse_raw",
    "mu_rmse_raw",
    "phi_mse_raw",
    "phi_rmse_raw",
    "gamma_mse_raw",
    "gamma_rmse_raw",

    # -----------------------------------------------------
    # Parameter scaled MSE/RMSE
    # -----------------------------------------------------
    "N0_mse_scaled",
    "N0_rmse_scaled",
    "mu_mse_scaled",
    "mu_rmse_scaled",
    "phi_mse_scaled",
    "phi_rmse_scaled",
    "gamma_mse_scaled",
    "gamma_rmse_scaled",

    # -----------------------------------------------------
    # Parameter MAE / MAPE / R2
    # -----------------------------------------------------
    "N0_mae_raw",
    "mu_mae_raw",
    "phi_mae_deg",
    "gamma_mae",

    "N0_mape_percent",
    "mu_mape_percent",
    "phi_mape_percent",
    "gamma_mape_percent",

    "N0_r2",
    "mu_r2",
    "phi_r2",
    "gamma_r2",

    "phi_within_tol",
    "gamma_within_tol",

    # -----------------------------------------------------
    # Latent lubrication raw metrics
    # -----------------------------------------------------
    "F_lub_mae_raw",
    "F_lub_mse_raw",
    "F_lub_rmse_raw",
    "F_lub_bias_raw",
    "F_lub_mape_percent",
    "F_lub_r2",
    "F_lub_target_mean",
    "F_lub_pred_mean",

    # -----------------------------------------------------
    # Latent lubrication log raw and scaled metrics
    # -----------------------------------------------------
    "log_F_lub_mse_raw",
    "log_F_lub_rmse_raw",
    "log_F_lub_r2_raw",

    "log_F_lub_scaled_mse",
    "log_F_lub_scaled_rmse",
    "log_F_lub_scaled_r2",

    "lub_supervised_weight_stage",
    "lub_supervised_raw_real",
    "lub_supervised_real",
    "total_contribution_lub_supervised_real",

    "lub_prior_weight_stage",
    "lub_prior_raw_real",
    "lub_prior_real",
    "total_contribution_lub_prior_real",
    "lub_prior_raw_rmse_standardized_log",
    "lub_prior_raw_rmse_log_unit",

    # -----------------------------------------------------
    # Force reconstruction raw and scaled metrics
    # -----------------------------------------------------
    "jacking_formula_mse_raw",
    "jacking_formula_rmse_raw",
    "jacking_formula_mse_scaled",
    "jacking_formula_rmse_scaled",
    "jacking_formula_mae",
    "jacking_formula_bias",
    "jacking_formula_mape_percent",
    "jacking_formula_r2",

    "friction_effective_mse_raw",
    "friction_effective_rmse_raw",
    "friction_effective_mse_scaled",
    "friction_effective_rmse_scaled",
    "friction_effective_mae",
    "friction_effective_bias",
    "friction_effective_mape_percent",
    "friction_effective_r2",

    # -----------------------------------------------------
    # Mu mean-collapse diagnostics
    # -----------------------------------------------------
    "mu_pred_mean",
    "mu_target_mean",
    "mu_pred_std",
    "mu_target_std",
    "mu_pred_abs_dist_from_train_mean",
    "mu_target_abs_dist_from_train_mean",
]

compact_cols = [
    col for col in compact_cols
    if col in stage_summary_df.columns
]

compact_real_summary_df = stage_summary_df[compact_cols].copy()



if len(missing_stages) > 0:
    missing_stage_df = pd.DataFrame({
        "missing_stage": missing_stages,
        "expected_checkpoint_path": [
            get_real_stage_checkpoint_path(stage)
            for stage in missing_stages
        ],
    })
else:
    missing_stage_df = pd.DataFrame()


def write_df_to_excel_safely(writer, df, sheet_name, index=False):
    max_excel_rows = 1_048_000

    if len(df) <= max_excel_rows:
        df.to_excel(
            writer,
            sheet_name=sheet_name[:31],
            index=index,
        )
        return

    n_chunks = int(np.ceil(len(df) / max_excel_rows))

    for chunk_idx in range(n_chunks):
        start = chunk_idx * max_excel_rows
        end = min((chunk_idx + 1) * max_excel_rows, len(df))

        chunk_sheet_name = f"{sheet_name[:27]}_{chunk_idx + 1}"

        df.iloc[start:end].to_excel(
            writer,
            sheet_name=chunk_sheet_name[:31],
            index=index,
        )

with pd.ExcelWriter(real_all_stage_output_path, engine="openpyxl") as writer:
    compact_real_summary_df.to_excel(
        writer,
        sheet_name="compact_summary",
        index=False,
    )

    stage_summary_df.to_excel(
        writer,
        sheet_name="stage_summary_all",
        index=False,
    )

    write_df_to_excel_safely(
        writer=writer,
        df=all_stage_predictions_df,
        sheet_name="all_stage_predictions",
        index=False,
    )

    if len(missing_stage_df) > 0:
        missing_stage_df.to_excel(
            writer,
            sheet_name="missing_stages",
            index=False,
        )


try:
    from openpyxl import load_workbook
    from openpyxl.styles import Font, Alignment
    from openpyxl.utils import get_column_letter

    wb = load_workbook(real_all_stage_output_path)

    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center")

        for col_idx, column_cells in enumerate(ws.columns, start=1):
            max_length = 0
            col_letter = get_column_letter(col_idx)

            for cell in column_cells:
                try:
                    cell_value = str(cell.value)
                    max_length = max(max_length, len(cell_value))
                except Exception:
                    pass

            adjusted_width = min(max(max_length + 2, 12), 35)
            ws.column_dimensions[col_letter].width = adjusted_width

    wb.save(real_all_stage_output_path)

except Exception as e:
    print("\nExcel formatting step was skipped because of this error:")
    print(e)



pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)

print("\n" + "=" * 120)
print("17 REAL-CASE ZERO-SHOT ALL-STAGE SUMMARY ")
print("=" * 120)
print(f"Stages requested: {REAL_STAGES_TO_TEST}")
print(f"Excel saved to:   {real_all_stage_output_path}")

if len(missing_stages) > 0:
    print(f"Missing stages skipped: {missing_stages}")

print("=" * 120)

print("\nCompact real-case all-stage comparison:")
print(compact_real_summary_df.round(6).to_string(index=False))
display(compact_real_summary_df.round(6))

print("\nFull stage summary:")
display(stage_summary_df.round(6))

print("\nPrediction preview:")
preview_cols = [
    "real_model_stage",
    "Jacking Distance",
    "Jacking Forces",
    "F_face",
    "True_F_fric_lub",
    "N0_pred",
    "mu_pred",
    "phi_deg_pred",
    "gamma_pred",
    "Actual_F_lub_pred",
]

for col in [
    "N0_teacher",
    "mu_teacher",
    "phi_deg_teacher",
    "gamma_teacher",
    "N0_target_ape_percent",
    "mu_target_ape_percent",
    "phi_deg_target_ape_percent",
    "gamma_target_ape_percent",
    "Actual_F_lub",
    "Actual_F_lub_ape_percent",
    "real_case_param_loss_row",
    "Actual_F_lub_abs_error_raw",
    "log_F_lub_scaled_sq_error",
    "Jacking_formula_ape_percent",
    "friction_effective_ape_percent",
]:
    if col in all_stage_predictions_df.columns:
        preview_cols.append(col)

preview_cols = [
    col for col in preview_cols
    if col in all_stage_predictions_df.columns
]

display(all_stage_predictions_df[preview_cols].head(20).round(6))

print("\n" + "=" * 120)
print("17 COMPLETE")
print("=" * 120)
