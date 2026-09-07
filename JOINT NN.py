import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# =========================================================
# 1. configuration
# =========================================================
SEED = int(os.environ.get("SEED", "42"))


DATA_FILE = os.environ.get(
    "DATA_FILE",
    "Phase1_BayesianAligned_V2.5_02_800Drives_StartLubAt80_01.xlsx",
)

SHEET_NAME = 0
SAVE_DIR = os.environ.get(
    "SAVE_DIR",
    "Ablation Study 1 Joint NN",
)

# Lubrication begins at 80 m.
START_DIST = 80.0

# "full_drive": use both unlubricated and lubricated portions.
# "lub_only":   use only rows at and after START_DIST.
DATA_COVERAGE_MODE = "lub_only"

if DATA_COVERAGE_MODE not in {"full_drive", "lub_only"}:
    raise ValueError(
        "DATA_COVERAGE_MODE must be either 'full_drive' or 'lub_only'."
    )

WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "50"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "512"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "50"))
PATIENCE = int(os.environ.get("PATIENCE", "20"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
DROPOUT = float(os.environ.get("DROPOUT", "0.10"))

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
LUB_EPS = 1e-3

# These are the only losses used for optimizer updates.
LAMBDA_TOTAL_FORCE = 1.0
LAMBDA_FRICTION = 1.0
LAMBDA_FACE = 1.0

INPUT_COLS = [
    "Jacking Forces",
    "Jacking Distance",
    "F_face",
    "True_F_fric_lub",
]

PARAM_COLS = [
    "N0",
    "mu",
    "phi_deg",
    "gamma",
]

PIPE_DIAMETER = 1.524
TBM_DIAMETER = 1.5748


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print("Using device:", device)

os.makedirs(SAVE_DIR, exist_ok=True)



# =========================================================
# 2. data preparation
# =========================================================
def load_dataset(path):
    if path is None:
        raise ValueError(
            "DATA_FILE is None. Please set DATA_FILE to your dataset path, "
            "for example: "
            "'Phase1_BayesianAligned_V2.5_02_800Drives_"
            "StartLubAt80_01.xlsx'"
        )

    path = str(path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file was not found: {path}\n"
            "Please make sure the Excel file is in the same folder as "
            "this script/notebook, or use the full file path."
        )

    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)

    elif path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(
            path,
            sheet_name=SHEET_NAME,
        )

    else:
        raise ValueError(
            f"Unsupported dataset format: {path}"
        )

    # Remove accidentally duplicated columns.
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Support datasets that name the friction angle "phi".
    if "phi" in df.columns and "phi_deg" not in df.columns:
        df = df.rename(
            columns={"phi": "phi_deg"}
        )

        valid_phi = df["phi_deg"].dropna()

        if (
            len(valid_phi) > 0
            and valid_phi.median() < 5.0
        ):
            df["phi_deg"] = np.degrees(
                df["phi_deg"]
            )

    # PARAM_COLS are retained only as synthetic evaluation labels.
    # The model and force losses receive only INPUT_COLS.
    required = (
        ["Drive_ID"]
        + INPUT_COLS
        + PARAM_COLS
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    # Retain only the columns used by this baseline.
    df = (
        df[required]
        .dropna()
        .sort_values(
            ["Drive_ID", "Jacking Distance"]
        )
        .reset_index(drop=True)
    )

    
    if DATA_COVERAGE_MODE == "lub_only":
        df = df[
            df["Jacking Distance"]
            >= START_DIST - 1e-9
        ].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No rows remain after applying "
            f"DATA_COVERAGE_MODE='{DATA_COVERAGE_MODE}'."
        )

    remaining_rows_per_drive = (
        df.groupby("Drive_ID")
        .size()
    )

    drives_with_insufficient_rows = (
        remaining_rows_per_drive[
            remaining_rows_per_drive < WINDOW_SIZE
        ]
        .index
        .tolist()
    )

    if len(drives_with_insufficient_rows) == len(
        remaining_rows_per_drive
    ):
        raise ValueError(
            "No drive contains enough retained rows to construct "
            f"a window of length {WINDOW_SIZE}."
        )

    print("\nDataset loading summary:")
    print(f"  File:               {path}")
    print(f"  Coverage mode:      {DATA_COVERAGE_MODE}")
    print(f"  Lubrication start:  {START_DIST:.1f} m")
    print(
        "  Retained distance: "
        f"{df['Jacking Distance'].min():.1f} to "
        f"{df['Jacking Distance'].max():.1f} m"
    )
    print(f"  Retained rows:      {len(df)}")
    print(
        "  Retained drives:    "
        f"{df['Drive_ID'].nunique()}"
    )

    if drives_with_insufficient_rows:
        print(
            "  Drives skipped during window construction: "
            f"{len(drives_with_insufficient_rows)}"
        )

    return df


def split_by_drive(df):
    drive_ids = np.array(
        sorted(df["Drive_ID"].unique())
    )

    rng = np.random.default_rng(SEED)
    rng.shuffle(drive_ids)

    n_train = int(
        len(drive_ids) * TRAIN_RATIO
    )

    n_val = int(
        len(drive_ids) * VAL_RATIO
    )

    train_ids = drive_ids[:n_train]

    val_ids = drive_ids[
        n_train:n_train + n_val
    ]

    test_ids = drive_ids[
        n_train + n_val:
    ]

    train_df = df[
        df["Drive_ID"].isin(train_ids)
    ].reset_index(drop=True)

    val_df = df[
        df["Drive_ID"].isin(val_ids)
    ].reset_index(drop=True)

    test_df = df[
        df["Drive_ID"].isin(test_ids)
    ].reset_index(drop=True)

    return train_df, val_df, test_df


class DriveWindowDataset(Dataset):
    def __init__(
        self,
        df,
        x_scaled,
        x_raw,
        y_raw,
    ):
        self.df = df.reset_index(drop=True)
        self.x_scaled = x_scaled
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.samples = []

        drive_values = self.df[
            "Drive_ID"
        ].values

        for drive_id in self.df[
            "Drive_ID"
        ].unique():
            indices = np.flatnonzero(
                drive_values == drive_id
            )

            if len(indices) < WINDOW_SIZE:
                continue

            for start in range(
                len(indices) - WINDOW_SIZE + 1
            ):
                window = indices[
                    start:start + WINDOW_SIZE
                ]

                target = window[-1]

                self.samples.append(
                    (window, target)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        window, target = self.samples[index]

        return (
            torch.tensor(
                self.x_scaled[window],
                dtype=torch.float32,
            ),
            torch.tensor(
                self.x_raw[target],
                dtype=torch.float32,
            ),
            torch.tensor(
                self.y_raw[target],
                dtype=torch.float32,
            ),
        )




# =========================================================
# 3. simple CNN: one output head only
# =========================================================
class SimpleCNNInverseModel(nn.Module):
    def __init__(self, input_features):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Dropout(DROPOUT) if DROPOUT > 0 else nn.Identity(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.output_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )

    def forward(self, x_seq):
        z = self.features(x_seq.permute(0, 2, 1))
        z = self.pool(z).squeeze(-1)

        raw_output = self.output_head(z)

        
        params_raw = torch.nn.functional.softplus(raw_output[:, :4]) + 1e-6

        
        lub_raw = torch.nn.functional.softplus(raw_output[:, 4:5]) + LUB_EPS
        log_lub_raw = torch.log(lub_raw)

        return params_raw, log_lub_raw, lub_raw




# =========================================================
# 4. physics-only force reconstruction
# =========================================================
def reconstruct_forces(params_raw, lub_pred_raw, x_target_raw):
    

    N0 = params_raw[:, 0:1]
    mu = params_raw[:, 1:2]
    phi_deg = params_raw[:, 2:3]
    gamma = params_raw[:, 3:4]

    phi_rad = phi_deg * torch.pi / 180.0
    tan_phi = torch.tan(phi_rad)
    tan_phi = torch.where(
        torch.abs(tan_phi) < 1e-6,
        torch.full_like(tan_phi, 1e-6),
        tan_phi,
    )

    f_face = 10.0 * 1.32 * torch.pi * TBM_DIAMETER * N0
    face_pred = f_face * torch.pi * (TBM_DIAMETER / 2.0) ** 2

    f_fric = (
        mu
        * gamma
        * PIPE_DIAMETER
        * torch.cos(torch.pi / 4.0 + phi_rad / 2.0)
        / (2.0 * tan_phi)
    )

    k_fric_pred = f_fric * torch.pi * PIPE_DIAMETER

    jack_distance = x_target_raw[:, 1:2]
    friction_native_pred = k_fric_pred * jack_distance

    lub_effective = torch.minimum(lub_pred_raw, friction_native_pred)
    friction_pred = friction_native_pred - lub_effective
    total_pred = face_pred + friction_pred

    return total_pred, friction_pred, face_pred, k_fric_pred, friction_native_pred


def parameter_metrics(params_raw, y_raw, param_eval_scale):
    error = params_raw - y_raw

    mse_raw_cols = torch.mean(error.pow(2), dim=0)
    mse_scaled_cols = torch.mean(
        (error / param_eval_scale.view(1, -1)).pow(2),
        dim=0,
    )
    mae_cols = torch.mean(torch.abs(error), dim=0)

    eps = 1e-8
    mape_cols = torch.mean(torch.abs(error / torch.clamp(torch.abs(y_raw), min=eps)), dim=0) * 100.0

    return {
        "param_mse_raw": torch.mean(mse_raw_cols),
        "param_mse_scaled": torch.mean(mse_scaled_cols),
        "param_mae": torch.mean(mae_cols),

        "mse_N0_raw": mse_raw_cols[0],
        "mse_mu_raw": mse_raw_cols[1],
        "mse_phi_raw": mse_raw_cols[2],
        "mse_gamma_raw": mse_raw_cols[3],

        "mse_N0_scaled": mse_scaled_cols[0],
        "mse_mu_scaled": mse_scaled_cols[1],
        "mse_phi_scaled": mse_scaled_cols[2],
        "mse_gamma_scaled": mse_scaled_cols[3],

        "mae_N0": mae_cols[0],
        "mae_mu": mae_cols[1],
        "mae_phi": mae_cols[2],
        "mae_gamma": mae_cols[3],

        "mape_N0": mape_cols[0],
        "mape_mu": mape_cols[1],
        "mape_phi": mape_cols[2],
        "mape_gamma": mape_cols[3],
    }


def run_epoch(
    model,
    loader,
    total_scale,
    friction_scale,
    face_scale,
    param_eval_scale,
    optimizer=None,
):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    keys = [
        "total",
        "total_force_loss",
        "friction_loss",
        "face_loss",
        "force_total_mae",
        "force_friction_mae",

        "param_mse_raw",
        "param_mse_scaled",
        "param_mae",

        "mse_N0_raw",
        "mse_mu_raw",
        "mse_phi_raw",
        "mse_gamma_raw",

        "mse_N0_scaled",
        "mse_mu_scaled",
        "mse_phi_scaled",
        "mse_gamma_scaled",

        "mae_N0",
        "mae_mu",
        "mae_phi",
        "mae_gamma",

        "mape_N0",
        "mape_mu",
        "mape_phi",
        "mape_gamma",
    ]

    totals = {key: 0.0 for key in keys}
    n_samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for x_seq, x_raw, y_raw in loader:
            x_seq = x_seq.to(device)
            x_raw = x_raw.to(device)
            y_raw = y_raw.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            params_pred, _log_lub_pred, lub_pred = model(x_seq)

            total_pred, friction_pred, face_pred, _k_pred, _native_pred = reconstruct_forces(
                params_pred,
                lub_pred,
                x_raw,
            )

            total_obs = x_raw[:, 0:1]
            face_obs = x_raw[:, 2:3]
            friction_obs = x_raw[:, 3:4]

            total_force_loss = torch.mean(((total_pred - total_obs) / total_scale).pow(2))
            friction_loss = torch.mean(((friction_pred - friction_obs) / friction_scale).pow(2))
            face_loss = torch.mean(((face_pred - face_obs) / face_scale).pow(2))

            force_loss = (
                LAMBDA_TOTAL_FORCE * total_force_loss
                + LAMBDA_FRICTION * friction_loss
                + LAMBDA_FACE * face_loss
            )

            if is_train:
                force_loss.backward()
                optimizer.step()

            with torch.no_grad():
                metric_values = parameter_metrics(
                    params_pred,
                    y_raw,
                    param_eval_scale,
                )

                values = {
                    "total": force_loss.detach(),
                    "total_force_loss": total_force_loss.detach(),
                    "friction_loss": friction_loss.detach(),
                    "face_loss": face_loss.detach(),
                    "force_total_mae": torch.mean(torch.abs(total_pred - total_obs)).detach(),
                    "force_friction_mae": torch.mean(torch.abs(friction_pred - friction_obs)).detach(),
                    **metric_values,
                }

            batch_size = x_seq.size(0)

            for key, value in values.items():
                totals[key] += value.item() * batch_size

            n_samples += batch_size

    if n_samples == 0:
        raise ValueError(
            "No windows were created. Please check WINDOW_SIZE and the number of rows per Drive_ID."
        )

    return {key: value / n_samples for key, value in totals.items()}





# =========================================================
# 5. main training workflow
# =========================================================
df = load_dataset(DATA_FILE)
train_df, val_df, test_df = split_by_drive(df)

scaler_x = StandardScaler()


def arrays(frame):
    x = frame[INPUT_COLS].values.astype(np.float32)
    y = frame[PARAM_COLS].values.astype(np.float32)
    return x, y


x_train, y_train = arrays(train_df)
x_val, y_val = arrays(val_df)
x_test, y_test = arrays(test_df)

x_train_scaled = scaler_x.fit_transform(x_train)
x_val_scaled = scaler_x.transform(x_val)
x_test_scaled = scaler_x.transform(x_test)

train_dataset = DriveWindowDataset(
    train_df,
    x_train_scaled,
    x_train,
    y_train,
)

val_dataset = DriveWindowDataset(
    val_df,
    x_val_scaled,
    x_val,
    y_val,
)

test_dataset = DriveWindowDataset(
    test_df,
    x_test_scaled,
    x_test,
    y_test,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

total_scale = torch.tensor(
    np.std(x_train[:, 0]) + 1e-6,
    dtype=torch.float32,
    device=device,
)

friction_scale = torch.tensor(
    np.std(x_train[:, 3]) + 1e-6,
    dtype=torch.float32,
    device=device,
)

face_scale = torch.tensor(
    np.std(x_train[:, 2]) + 1e-6,
    dtype=torch.float32,
    device=device,
)


param_eval_scale = torch.tensor(
    np.std(y_train, axis=0, ddof=0) + 1e-8,
    dtype=torch.float32,
    device=device,
)

model = SimpleCNNInverseModel(
    input_features=len(INPUT_COLS),
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=10,
)

best_path = os.path.join(SAVE_DIR, "best_simple_cnn_force_loss_only.pt")
history = []

best_val = float("inf")
bad_epochs = 0

print("=" * 80)
print("Simple CNN Baseline: Force-Loss Only")
print("=" * 80)
print(f"Device: {device}")
print(f"Dataset: {DATA_FILE}")
print(f"Rows: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
print(f"Drives: train={train_df['Drive_ID'].nunique()}, val={val_df['Drive_ID'].nunique()}, test={test_df['Drive_ID'].nunique()}")
print(f"Windows: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
print(f"Window size: {WINDOW_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Weight decay: {WEIGHT_DECAY}")
print(f"Save directory: {SAVE_DIR}")
print(f"Total force scale: {total_scale.item():.6f}")
print(f"Friction scale: {friction_scale.item():.6f}")
print(f"Face scale: {face_scale.item():.6f}")
print("Evaluation-only parameter stds:", param_eval_scale.detach().cpu().numpy())
print("-" * 80)
print("Baseline: one CNN + one output head; total-force, friction, and face losses backpropagate.")
print("True parameter labels are evaluation-only; Actual_F_lub is not loaded or used.")
print("No parameter-label scaling, site-specific parameter bounds, or lubrication prior is used.")
print("=" * 80)




# =========================================================
# 6. training loop
# =========================================================
for epoch in range(1, NUM_EPOCHS + 1):
    train_metrics = run_epoch(
        model,
        train_loader,
        total_scale,
        friction_scale,
        face_scale,
        param_eval_scale,
        optimizer=optimizer,
    )

    val_metrics = run_epoch(
        model,
        val_loader,
        total_scale,
        friction_scale,
        face_scale,
        param_eval_scale,
    )

    scheduler.step(val_metrics["total"])

    row = {"epoch": epoch}
    row.update({f"train_{key}": value for key, value in train_metrics.items()})
    row.update({f"val_{key}": value for key, value in val_metrics.items()})
    row["lr"] = optimizer.param_groups[0]["lr"]
    history.append(row)

    improved = val_metrics["total"] < best_val

    if improved:
        best_val = val_metrics["total"]
        bad_epochs = 0
        torch.save(model.state_dict(), best_path)
        save_mark = " *best"
    else:
        bad_epochs += 1
        save_mark = ""

    print(
        f"Ep [{epoch:04d}/{NUM_EPOCHS}] | "
        f"Force: {train_metrics['total']:.6f}/{val_metrics['total']:.6f} | "
        f"TotF: {train_metrics['total_force_loss']:.6f}/{val_metrics['total_force_loss']:.6f} | "
        f"Fric: {train_metrics['friction_loss']:.6f}/{val_metrics['friction_loss']:.6f} | "
        f"Face: {train_metrics['face_loss']:.6f}/{val_metrics['face_loss']:.6f} | "
        f"MSE(N0 scaled eval): {train_metrics['mse_N0_scaled']:.6f}/{val_metrics['mse_N0_scaled']:.6f} | "
        f"MSE(mu scaled eval): {train_metrics['mse_mu_scaled']:.6f}/{val_metrics['mse_mu_scaled']:.6f} | "
        f"MSE(mu raw eval): {train_metrics['mse_mu_raw']:.6f}/{val_metrics['mse_mu_raw']:.6f} | "
        f"MAE(phi eval): {train_metrics['mae_phi']:.6f}/{val_metrics['mae_phi']:.6f} | "
        f"MAPE(phi eval): {train_metrics['mape_phi']:.3f}%/{val_metrics['mape_phi']:.3f}% | "
        f"MAE(gamma eval): {train_metrics['mae_gamma']:.6f}/{val_metrics['mae_gamma']:.6f} | "
        f"MAPE(gamma eval): {train_metrics['mape_gamma']:.3f}%/{val_metrics['mape_gamma']:.3f}% | "
        f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        f"{save_mark}"
    )
    if bad_epochs >= PATIENCE:
        print("-" * 80)
        print(f"Early stopping at epoch {epoch}.")
        print(f"Best validation force loss: {best_val:.6f}")
        print("-" * 80)
        break




# =========================================================
# 7. final test evaluation
# =========================================================
print("\nLoading best model for final test evaluation...")
model.load_state_dict(
    torch.load(best_path, map_location=device)
)

test_metrics = run_epoch(
    model=model,
    loader=test_loader,
    total_scale=total_scale,
    friction_scale=friction_scale,
    face_scale=face_scale,
    param_eval_scale=param_eval_scale,
    optimizer=None,
)

history_path = os.path.join(
    SAVE_DIR,
    "training_history.csv",
)

metrics_path = os.path.join(
    SAVE_DIR,
    "test_metrics.json",
)

pd.DataFrame(history).to_csv(
    history_path,
    index=False,
)

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(test_metrics, file, indent=2)

print("\n" + "=" * 80)
print("TRAINING COMPLETE: SIMPLE CNN FORCE-LOSS-ONLY BASELINE")
print("=" * 80)
print(f"Best model file:                   {best_path}")
print(f"Best validation force loss:        {best_val:.6f}")
print(f"Training history file:             {history_path}")
print(f"Test metrics file:                 {metrics_path}")

print("-" * 80)
print("Test reconstruction metrics:")
print(
    f"{'Total force loss':>34s}: "
    f"{test_metrics['total']:.6f}"
)
print(
    f"{'Total-force reconstruction':>34s}: "
    f"{test_metrics['total_force_loss']:.6f}"
)
print(
    f"{'Friction reconstruction':>34s}: "
    f"{test_metrics['friction_loss']:.6f}"
)
print(
    f"{'Face-resistance reconstruction':>34s}: "
    f"{test_metrics['face_loss']:.6f}"
)
print(
    f"{'Total-force MAE':>34s}: "
    f"{test_metrics['force_total_mae']:.6f}"
)
print(
    f"{'Friction MAE':>34s}: "
    f"{test_metrics['force_friction_mae']:.6f}"
)

print("-" * 80)
print("Test parameter MSE (evaluation only):")
print(
    f"{'Parameter':<12s}"
    f"{'Scaled MSE':>18s}"
    f"{'Raw MSE':>18s}"
)
print("-" * 48)

for name in ["N0", "mu", "phi", "gamma"]:
    print(
        f"{name:<12s}"
        f"{test_metrics[f'mse_{name}_scaled']:>18.6f}"
        f"{test_metrics[f'mse_{name}_raw']:>18.6f}"
    )

print("-" * 80)
print("Test parameter MAE and MAPE (evaluation only):")
print(
    f"{'Parameter':<12s}"
    f"{'MAE':>18s}"
    f"{'MAPE (%)':>18s}"
)
print("-" * 48)

for name in ["N0", "mu", "phi", "gamma"]:
    print(
        f"{name:<12s}"
        f"{test_metrics[f'mae_{name}']:>18.6f}"
        f"{test_metrics[f'mape_{name}']:>18.3f}"
    )

print("-" * 80)
print("All test metrics:")

for key, value in test_metrics.items():
    print(f"{key:>34s}: {value:.6f}")

print("=" * 80)

