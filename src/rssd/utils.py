import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


def seed_everything(seed=42):
    """Set random seed for all libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to: {seed}")


# -------------------------
# simple Data replacement
# -------------------------
class Data:
    def __init__(self, x, edge_index):
        self.x = x
        self.edge_index = edge_index

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        return self


class WindowDataset(Dataset):
    def __init__(self, X, y, edge_index):
        """
        X: [N, days_x, num_nodes, F]
        y: [N, num_nodes, days_y]
        """
        self.X, self.y = X, y
        self.edge_index = edge_index

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        graphs = []
        for t in range(self.X.shape[1]):
            x_t = self.X[idx, t]  # [num_nodes, F]
            graphs.append(Data(x=x_t, edge_index=self.edge_index))
        target = self.y[idx]  # [num_nodes, days_y]
        return graphs, target


def build_supervised_split_from_reservoir_blocks(
    scaler_data,
    reservoir_names_in_node_order,
    split,
    purge_head_windows=0,
):
    """Assemble one chronological split into the WindowDataset tensor layout.

    ``all_rsr_data_<scaler_type>.pkl`` stores the training, validation and test
    arrays separately for each reservoir.  This helper stacks one of those blocks
    across reservoirs without changing node order.

    ``purge_head_windows`` removes the first windows from validation or test.
    With overlapping windows, setting it to ``Tin + horizon - 1`` establishes
    an embargo after the preceding split while keeping all training windows
    used to fit the stored scalers.
    """
    split = str(split).lower().strip()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split={split!r}; expected train, val, or test.")
    purge_head_windows = int(purge_head_windows)
    if purge_head_windows < 0:
        raise ValueError("purge_head_windows must be non-negative.")
    if split == "train" and purge_head_windows:
        raise ValueError("Training windows must not be head-purged.")
    names = [str(name) for name in reservoir_names_in_node_order]
    if not names:
        raise ValueError("reservoir_names_in_node_order must not be empty.")

    x_blocks = []
    y_blocks = []
    for name in names:
        if name not in scaler_data:
            raise KeyError(f"Reservoir {name!r} missing from scaler_data.")
        reservoir_block = scaler_data[name]
        if split not in reservoir_block:
            raise KeyError(f"Split {split!r} missing for reservoir {name!r}.")
        block = reservoir_block[split]
        if not isinstance(block, dict) or "X" not in block or "y" not in block:
            raise TypeError(
                f"Expected scaler_data[{name!r}][{split!r}] to contain X and y."
            )
        x = np.asarray(block["X"], dtype=np.float32)
        y = np.asarray(block["y"], dtype=np.float32)
        if x.ndim != 3 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Malformed {name}/{split} arrays: X={x.shape}, y={y.shape}."
            )
        valid = ~(np.isnan(x).any(axis=(1, 2)) | np.isnan(y).any(axis=1))
        x = x[valid]
        y = y[valid]
        if x.shape[0] == 0:
            raise ValueError(f"No finite samples remain for {name}/{split}.")
        x_blocks.append(x)
        y_blocks.append(y)

    min_length = min(x.shape[0] for x in x_blocks)
    start_index = 0
    if split in {"val", "test"} and purge_head_windows:
        if min_length <= purge_head_windows:
            raise ValueError(
                f"Cannot purge {purge_head_windows} windows from {split} "
                f"with aligned length {min_length}."
            )
        start_index = purge_head_windows
    x_blocks = [x[start_index:min_length] for x in x_blocks]
    y_blocks = [y[start_index:min_length] for y in y_blocks]

    # Per-reservoir X is [samples, input_days, features].  Insert nodes as
    # dimension 2 to match [samples, input_days, nodes, features].
    X = torch.from_numpy(np.stack(x_blocks, axis=2))
    # Per-reservoir y is [samples, forecast_days].
    y = torch.from_numpy(np.stack(y_blocks, axis=1))
    return X, y


# -------------------------
# transforms for inflow
# -------------------------
def signed_expm1(x: np.ndarray) -> np.ndarray:
    """Inverse of signed_log1p: sign(x)*(exp(|x|)-1)."""
    return np.sign(x) * (np.expm1(np.abs(x)))


def _build_idx_to_reservoir_strict(encode_map, n_nodes: int):
    """
    Support two formats:
      A) {reservoir_name(str): node_idx(int)}
      B) {node_idx(int): reservoir_name(str)}
    Return: {0..n_nodes-1 -> reservoir_name(str)}
    """
    if encode_map is None:
        raise ValueError("encode_map is required for local inverse transform (strict mode).")
    if not isinstance(encode_map, dict) or len(encode_map) == 0:
        raise ValueError("encode_map must be a non-empty dict.")

    k0, v0 = next(iter(encode_map.items()))

    idx_to_res = {}

    # A) name -> idx
    if isinstance(k0, str) and isinstance(v0, (int, np.integer)):
        for name, idx in encode_map.items():
            idx = int(idx)
            if 0 <= idx < n_nodes:
                idx_to_res[idx] = str(name)

    # B) idx -> name
    elif isinstance(k0, (int, np.integer)) and isinstance(v0, str):
        for idx, name in encode_map.items():
            idx = int(idx)
            if 0 <= idx < n_nodes:
                idx_to_res[idx] = str(name)

    else:
        raise ValueError(f"Unrecognized encode_map format: key={type(k0)}, val={type(v0)}")

    # strict: must cover every node index
    missing = [i for i in range(n_nodes) if i not in idx_to_res]
    if missing:
        raise RuntimeError(
            f"encode_map does not cover all node indices 0..{n_nodes-1}. "
            f"missing count={len(missing)} example={missing[:10]}"
        )
    return idx_to_res


def inverse_transform_predictions(predictions, targets, scaler_data, encode_map=None):
    """
    Unified inverse transform for both global/local scalers.
    Supports optional y_transform inversion (e.g., signed_log1p).
    STRICT for local: requires correct idx<->reservoir alignment.
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    if predictions.ndim != 3:
        raise ValueError(f"predictions must be (n_samples,n_nodes,n_days), got {predictions.shape}")
    if targets.shape != predictions.shape:
        raise ValueError(f"targets shape {targets.shape} != predictions shape {predictions.shape}")

    n_samples, n_nodes, n_days = predictions.shape
    scaler_type = scaler_data.get("params", {}).get("scaler_type", "global")
    y_transform = scaler_data.get("params", {}).get("y_transform", "none")

    if scaler_type == "global":
        scaler_y = scaler_data["scaler_y"]
        pred_inv = scaler_y.inverse_transform(predictions.reshape(-1, 1)).reshape(n_samples, n_nodes, n_days)
        targ_inv = scaler_y.inverse_transform(targets.reshape(-1, 1)).reshape(n_samples, n_nodes, n_days)

    elif scaler_type == "local":
        local_scalers_y = scaler_data["local_scalers_y"]
        idx_to_reservoir = _build_idx_to_reservoir_strict(encode_map, n_nodes)

        pred_inv = np.zeros_like(predictions, dtype=np.float32)
        targ_inv = np.zeros_like(targets, dtype=np.float32)

        # strict: every reservoir name must have a scaler
        missing_scalers = [idx_to_reservoir[i] for i in range(n_nodes) if idx_to_reservoir[i] not in local_scalers_y]
        if missing_scalers:
            raise RuntimeError(f"Missing local y scalers for reservoirs: {missing_scalers[:10]} (count={len(missing_scalers)})")

        for node_idx in range(n_nodes):
            reservoir_name = idx_to_reservoir[node_idx]
            scaler_y = local_scalers_y[reservoir_name]

            p = predictions[:, node_idx, :].reshape(-1, 1)
            t = targets[:, node_idx, :].reshape(-1, 1)

            # for a MinMaxScaler, clip the scaled input to feature_range first to avoid
            # extrapolating outside the fitted range
            if hasattr(scaler_y, "feature_range"):
                lo, hi = scaler_y.feature_range
                # Only clip predictions; never clip targets (ground truth).
                # p = np.clip(p, lo, hi)

                # Optional: leave targets as-is to reflect real OOB events in test.
                # t = np.clip(t, lo, hi)  # <-- DELETE this line

            p_inv = scaler_y.inverse_transform(p).reshape(n_samples, n_days)
            t_inv = scaler_y.inverse_transform(t).reshape(n_samples, n_days)

            pred_inv[:, node_idx, :] = p_inv
            targ_inv[:, node_idx, :] = t_inv

    else:
        raise ValueError(f"Invalid scaler_type: {scaler_type}. Must be 'global' or 'local'.")

    # invert robust transform if enabled
    if y_transform == "log1p":
        pred_inv = np.expm1(pred_inv)
        targ_inv = np.expm1(targ_inv)
    elif y_transform == "signed_log1p":
        pred_inv = signed_expm1(pred_inv)
        targ_inv = signed_expm1(targ_inv)
    
    return pred_inv, targ_inv


def load_preprocessed_data(data_path, scaler_type="global"):
    """
    Load:
      - scaler_data from all_rsr_data_<scaler_type>.pkl
      - supervised_data from _GNN_supervise_<scaler_type>.pt
    Compatible with pt structure:
      {"graph_data":..., "supervised_data": {...}}
    """
    import pickle

    parsed_path = os.path.join(data_path, "parsed")

    all_rsr_data_file = os.path.join(parsed_path, f"all_rsr_data_{scaler_type}.pkl")
    if not os.path.exists(all_rsr_data_file):
        raise FileNotFoundError(f"Not found: {all_rsr_data_file}. Run _preprocess.py first.")
    with open(all_rsr_data_file, "rb") as f:
        all_rsr_data = pickle.load(f)

    scaler_data = {
        "scaler_X": all_rsr_data.get("scaler_X"),
        "scaler_y": all_rsr_data.get("scaler_y"),
        "local_scalers_X": all_rsr_data.get("local_scalers_X"),
        "local_scalers_y": all_rsr_data.get("local_scalers_y"),
        "params": all_rsr_data.get("params", {}),
        "diagnostics": all_rsr_data.get("diagnostics", {}),
    }

    supervised_file = os.path.join(parsed_path, f"_GNN_supervise_{scaler_type}.pt")
    if not os.path.exists(supervised_file):
        raise FileNotFoundError(f"Not found: {supervised_file}. Run _preprocess.py first.")

    sup_raw = torch.load(supervised_file, map_location="cpu", weights_only=False)
    if isinstance(sup_raw, dict) and "supervised_data" in sup_raw:
        supervised_data = sup_raw["supervised_data"]
    else:
        supervised_data = sup_raw

    return scaler_data, supervised_data


def check_available_data_files(data_path):
    parsed_path = os.path.join(data_path, "parsed")
    available = {}
    for scaler_type in ["global", "local"]:
        scaler_file = os.path.join(parsed_path, f"all_rsr_data_{scaler_type}.pkl")
        supervised_file = os.path.join(parsed_path, f"_GNN_supervise_{scaler_type}.pt")
        available[scaler_type] = {
            "scaler_data": os.path.exists(scaler_file),
            "supervised_data": os.path.exists(supervised_file),
            "complete": os.path.exists(scaler_file) and os.path.exists(supervised_file),
        }
    return available


def create_logging_directory(model_name, scaler_type="global"):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    log_dir = os.path.join("logs", model_name)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir, timestamp


def collate_zip(batch):
    # keep contract: _collate = lambda b: list(zip(*b))
    return list(zip(*batch))


def save_training_results(log_dir, results_text, timestamp):
    results_file = os.path.join(log_dir, f"results_{timestamp}.txt")
    with open(results_file, "w") as f:
        f.write(results_text)
    print(f"Training results saved to: {results_file}")


def save_best_checkpoint(log_dir, model, optimizer, epoch, loss, timestamp):
    checkpoint_file = os.path.join(log_dir, f"checkpoint_{timestamp}.pth")
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "timestamp": timestamp,
    }
    torch.save(checkpoint, checkpoint_file)
    print(f"Best checkpoint saved to: {checkpoint_file}")


class TrainingLogger:
    def __init__(
        self,
        model_name,
        scaler_type="global",
        use_pretrain=False,
        log_dir_override=None,
        timestamp_override=None,
    ):
        self.model_name = model_name
        self.scaler_type = scaler_type
        self.use_pretrain = use_pretrain

        if (log_dir_override is None) and (timestamp_override is None):
            # backward-compatible old behavior
            self.log_dir, self.timestamp = create_logging_directory(model_name, scaler_type)
        else:
            from datetime import datetime

            self.log_dir = str(log_dir_override) if log_dir_override is not None else os.path.join("logs", model_name)
            os.makedirs(self.log_dir, exist_ok=True)

            self.timestamp = (
                str(timestamp_override)
                if timestamp_override is not None
                else datetime.now().strftime("%Y%m%d%H%M")
            )

        if use_pretrain:
            self.timestamp = self.timestamp + "_p"

        self.best_val_loss = float("inf")
        self.training_logs = []

    def log_epoch(self, epoch, train_loss, val_loss, lr=None):
        log_entry = f"Epoch {epoch:3d}  "
        if lr is not None:
            log_entry += f"Learning Rate: {lr:.6f}  "
        log_entry += f"Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}"
        self.training_logs.append(log_entry)
        print(log_entry)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            return True
        return False

    def save_checkpoint(self, model, optimizer, epoch, loss):
        save_best_checkpoint(self.log_dir, model, optimizer, epoch, loss, self.timestamp)

    def save_results(self, additional_info=""):
        results_text = f"Training Results for {self.model_name}\n"
        results_text += f"Timestamp: {self.timestamp}\n"
        results_text += f"Best Validation Loss: {self.best_val_loss:.6f}\n"
        results_text += "=" * 50 + "\n"
        results_text += "\n".join(self.training_logs)

        if additional_info:
            results_text += "\n" + "=" * 50 + "\n"
            results_text += additional_info

        save_training_results(self.log_dir, results_text, self.timestamp)


def _adjust_checkpoint_time(model_name, scaler_type, pretrain_time):
    import re
    checkpoint_path = os.path.join("logs", model_name, scaler_type, f"checkpoint_{pretrain_time}.pth")
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at specified time: {pretrain_time}")
        return pretrain_time

    checkpoint_dir = os.path.join("logs", model_name, scaler_type)
    if not os.path.exists(checkpoint_dir):
        print(f"Warning: Checkpoint directory does not exist: {checkpoint_dir}")
        return None

    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith("checkpoint_") and file.endswith(".pth"):
            match = re.search(r"checkpoint_(\d+)\.pth", file)
            if match:
                checkpoint_files.append(match.group(1))

    if not checkpoint_files:
        print(f"Warning: No checkpoint files found in {checkpoint_dir}")
        return None

    checkpoint_files.sort()
    earliest_time = checkpoint_files[0]
    print(f"Original pretrain_time '{pretrain_time}' not found. Using earliest checkpoint: {earliest_time}")
    return earliest_time
