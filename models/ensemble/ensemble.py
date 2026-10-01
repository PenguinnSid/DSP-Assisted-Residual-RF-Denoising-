"""
Ensemble of the three denoisers (AutoEncoder, CNN, LSTM) using their *_best.pt checkpoints.

All three models take  (N, seq_len, 2)  [real, imag]  and return  (N, seq_len, 2),
so their outputs can be combined element-wise.

Combine modes:
    "mean"      simple average of the three outputs
    "median"    element-wise median (robust if one model is off on a sample)
    "weighted"  weights proportional to 1 / validation-MSE of each model
                (call ensemble.fit_weights(X_val, y_val) first)
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

MODELS_DIR = Path(__file__).resolve().parent.parent  # -> models/

# name -> folder, class name in that folder's model.py, constructor kwargs, checkpoint file
REGISTRY = {
    "AutoEncoder": dict(folder="AutoEncoder", cls="Autoencoder", kwargs={}, ckpt="ae_v1_best.pt"),
    "CNN":         dict(folder="CNN",         cls="CNN",         kwargs={}, ckpt="cnn_v1_best.pt"),
    # dropout_rate has no effect in eval(); state_dict keys are identical to training.
    "LSTM":        dict(folder="LSTM",        cls="LSTM",
                        kwargs=dict(input_size=2, hidden_size=64, num_layers=2,
                                    output_size=2, dropout_rate=0.0),
                        ckpt="lstm_v1_best.pt"),
}


def _load_class(folder, cls_name):
    """Import <folder>/model.py by file path. All three files are called model.py,
    so a normal `from model import ...` would collide; give each a unique module name."""
    path = MODELS_DIR / folder / "model.py"
    mod_name = f"_ens_{folder.lower()}_model"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return getattr(module, cls_name)


def load_member(name, device):
    cfg = REGISTRY[name]
    ckpt_path = MODELS_DIR / cfg["folder"] / "checkpoints" / cfg["ckpt"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run {cfg['folder']}/train.py first.")
    model = _load_class(cfg["folder"], cfg["cls"])(**cfg["kwargs"])
    # your train.py scripts save plain state_dicts
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device).eval()


class DenoiserEnsemble:
    def __init__(self, device=None, members=None, mode="mean"):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.names = list(members or REGISTRY)
        self.models = {n: load_member(n, self.device) for n in self.names}
        self.mode = mode
        self.weights = {n: 1.0 / len(self.names) for n in self.names}  # overwritten by fit_weights

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_members(self, X, batch_size=64):
        """X: (N, seq_len, 2) float32 numpy. Returns {name: (N, seq_len, 2) numpy}."""
        out = {n: [] for n in self.names}
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(self.device)
            for n, m in self.models.items():
                out[n].append(m(xb).cpu().numpy())
        return {n: np.concatenate(v, axis=0) for n, v in out.items()}

    def combine(self, member_preds, mode=None):
        """Combine already-computed member predictions (lets you compare modes without re-running the nets)."""
        mode = mode or self.mode
        stack = np.stack([member_preds[n] for n in self.names], axis=0)  # (M, N, L, 2)
        if mode == "mean":
            return stack.mean(axis=0)
        if mode == "median":
            return np.median(stack, axis=0)
        if mode == "weighted":
            w = np.array([self.weights[n] for n in self.names], dtype=np.float32)
            return np.tensordot(w / w.sum(), stack, axes=(0, 0))
        raise ValueError(f"Unknown mode '{mode}' (use mean / median / weighted)")

    def predict(self, X, mode=None, batch_size=64):
        return self.combine(self.predict_members(X, batch_size), mode)

    # ------------------------------------------------------------------ weights
    def fit_weights(self, X_val, y_val, batch_size=64):
        """Inverse-validation-MSE weights. Use the validation split, never test."""
        preds = self.predict_members(X_val, batch_size)
        mse = {n: float(np.mean((preds[n] - y_val) ** 2)) for n in self.names}
        inv = {n: 1.0 / v for n, v in mse.items()}
        total = sum(inv.values())
        self.weights = {n: v / total for n, v in inv.items()}
        return self.weights, mse


def build_ensemble(mode="mean", device=None):
    return DenoiserEnsemble(device=device, mode=mode)