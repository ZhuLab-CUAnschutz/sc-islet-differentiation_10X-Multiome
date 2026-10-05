"""Model builder for the seq->expression arms.

Two warm-starts x three regimes:
  arm    : 'decima'  -> Decima trunk (already expression-trained) + fresh 22-task head
           'borzoi'  -> Borzoi trunk + gene-mask channel + fresh 22-task head
  regime : 'probe'   -> freeze trunk, train head only          (cheap diagnostic)
           'lora'    -> freeze trunk, train head + LoRA adapters (A30-friendly escalation)
           'full'    -> train everything                        (Decima's actual recipe)

NOTE (Decima Supp. Table 2): frozen-Borzoi + head-only ('borzoi','probe') is expected to
underperform; the ('decima','probe') rung is the promising cheap one. Escalate with 'lora'
on A30 24 GB, or 'full' with gradient checkpointing.

⚠️ Model loading, ConvHead, and the loss are decima/grelu API; VERIFY. LoRA is
implemented via `peft` (below) — functional, but confirm the discovered target-module
names match Borzoi's transformer layout on the installed version.
"""

import torch
import torch.nn as nn
import lightning as L

N_TASKS = 22


# ---------------------------------------------------------------- trunk + head
def _decima_trunk_and_head(n_tasks):
    """Pretrained Decima trunk (+gene-mask stem) with a fresh n_tasks ConvHead.

    VERIFY: load a single Decima replicate, keep its 1920-ch Borzoi trunk + the 5-ch
    stem, and swap ConvHead(8856) -> ConvHead(n_tasks, in_channels=1920, pool_func='avg').
    """
    from decima import load_decima_model  # type: ignore  # VERIFY
    from grelu.model.heads import ConvHead  # type: ignore  # VERIFY

    m = load_decima_model(ensemble=False)          # one replicate
    trunk = m.model.embedding                       # VERIFY attribute path to the trunk
    head = ConvHead(n_tasks=n_tasks, in_channels=1920, pool_func="avg")
    return trunk, head


def _borzoi_trunk_and_head(n_tasks):
    """Borzoi trunk widened to a 5th gene-mask channel + fresh n_tasks ConvHead.

    VERIFY: load Borzoi via grelu, then widen the stem conv 4->5 input channels
    (copy pretrained 4-ch weights, init the 5th fresh) exactly as Decima does.
    """
    from grelu.model.models import BorzoiModel  # type: ignore  # VERIFY
    from grelu.model.heads import ConvHead  # type: ignore  # VERIFY

    borzoi = BorzoiModel.load_from_pretrained("borzoi_human_fold0")  # VERIFY loader
    trunk = _add_gene_mask_channel(borzoi.embedding)                 # VERIFY stem widening
    head = ConvHead(n_tasks=n_tasks, in_channels=1920, pool_func="avg")
    return trunk, head


def _add_gene_mask_channel(trunk):
    """Widen the first conv from 4 to 5 input channels (Decima's gene-mask trick)."""
    stem = _first_conv(trunk)  # VERIFY: locate the stem conv module
    if stem.in_channels == 5:
        return trunk
    new = nn.Conv1d(5, stem.out_channels, stem.kernel_size, stride=stem.stride,
                    padding=stem.padding, bias=stem.bias is not None)
    with torch.no_grad():
        new.weight[:, :4] = stem.weight
        new.weight[:, 4:].zero_()
        if stem.bias is not None:
            new.bias.copy_(stem.bias)
    _replace_first_conv(trunk, new)  # VERIFY: set the module back
    return trunk


def _first_conv(module):  # VERIFY helper
    for m in module.modules():
        if isinstance(m, nn.Conv1d):
            return m
    raise RuntimeError("no Conv1d found in trunk")


def _replace_first_conv(module, new):  # VERIFY helper
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv1d):
            parent = module
            *path, last = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, last, new)
            return


def _apply_regime(trunk, head, regime, lora_rank=8, lora_conv=False):
    if regime == "probe":
        for p in trunk.parameters():
            p.requires_grad_(False)
    elif regime == "lora":
        for p in trunk.parameters():
            p.requires_grad_(False)
        _inject_lora(trunk, rank=lora_rank, include_conv=lora_conv)
    elif regime == "full":
        for p in trunk.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError(regime)
    for p in head.parameters():
        p.requires_grad_(True)


def _lora_target_modules(trunk, include_conv=False):
    """Discover the fully-qualified names of modules to adapt.

    Default: all nn.Linear (the attention q/k/v/proj + MLP projections in Borzoi's
    transformer blocks) — the highest-value, most portable LoRA target. Optionally add
    nn.Conv1d (Scooby also adapts conv, but peft's Conv1d support is version-dependent).
    """
    types = (nn.Linear,) if not include_conv else (nn.Linear, nn.Conv1d)
    names = [name for name, m in trunk.named_modules() if isinstance(m, types)]
    if not names:
        raise RuntimeError("no Linear/Conv1d modules found to adapt with LoRA")
    return names


def _inject_lora(trunk, rank=8, include_conv=False):
    """In-place LoRA injection via peft.

    Freezes the base trunk (already frozen by the caller) and adds rank-`rank` adapters
    to the discovered target modules; only the adapter (`lora_*`) params end up trainable,
    which configure_optimizers() then picks up alongside the head.
    """
    try:
        from peft import LoraConfig, inject_adapter_in_model  # type: ignore
    except ImportError as e:  # noqa: BLE001
        raise ImportError(
            "peft not installed. `pip install peft` (add it to scripts/setup/create_env.sh)."
        ) from e

    targets = _lora_target_modules(trunk, include_conv=include_conv)
    cfg = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,   # alpha = 2r is a common, stable default
        lora_dropout=0.05,
        bias="none",
        target_modules=targets,
    )
    inject_adapter_in_model(cfg, trunk)  # in-place; marks lora_* params trainable

    n_lora = sum(p.numel() for n, p in trunk.named_parameters() if "lora_" in n and p.requires_grad)
    print(f"LoRA r={rank} on {len(targets)} modules "
          f"({'Linear+Conv1d' if include_conv else 'Linear'}); trainable adapter params: {n_lora:,}")


# ---------------------------------------------------------------- lightning module
class SeqExpr(L.LightningModule):
    def __init__(self, arm="decima", regime="probe", lr=4e-5, n_tasks=N_TASKS,
                 lora_rank=8, lora_conv=False):
        super().__init__()
        self.save_hyperparameters()
        if arm == "decima":
            self.trunk, self.head = _decima_trunk_and_head(n_tasks)
        elif arm == "borzoi":
            self.trunk, self.head = _borzoi_trunk_and_head(n_tasks)
        else:
            raise ValueError(arm)
        _apply_regime(self.trunk, self.head, regime, lora_rank=lora_rank, lora_conv=lora_conv)

        # Decima's task-wise Poisson+multinomial loss (across the 22-track axis).
        from decima.model.loss import TaskWisePoissonMultinomialLoss  # type: ignore  # VERIFY

        self.loss_fn = TaskWisePoissonMultinomialLoss(total_weight=1e-4)

    def forward(self, x):
        # trunk -> (B, 1920, L'); head avg-pools -> (B, n_tasks); exp -> expression
        emb = self.trunk(x)
        return torch.exp(self.head(emb)).squeeze(-1)

    def _step(self, batch, stage):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=x.shape[0])
        return loss

    def training_step(self, b, _):
        return self._step(b, "train")

    def validation_step(self, b, _):
        return self._step(b, "val")

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.hparams.lr)
