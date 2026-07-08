"""The FL engine: one round loop that expresses all four methods.

    method       aggregate?  BN kept local?  pooled data?
    centralized      no            -              yes        <- ceiling
    local            no            -              no         <- floor
    fedavg          yes            no             no         <- global model  (H1, H2)
    fedbn           yes           yes             no         <- personalized  (H3)

Two invariants worth stating, because they are what make the comparison legitimate:

* **Identical init.** Every method starts from the same seeded random weights.
* **Matched compute.** Every hospital sees R*E local epochs, whether or not it federates. If
  local-only trained for fewer epochs, H1 ("collaboration helps") would just be measuring the
  longer training budget FedAvg received.

Evaluation happens **after aggregation and before the next round's local training** -- scoring the
true federated model (pure global for FedAvg; global body + own BN for FedBN) with no local-
adaptation contamination. Score after local training instead and FedAvg silently gains a round of
personalization, which is exactly what H2 claims it lacks.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import Config
from .data import load_index, select_cases
from .logging_utils import MetricsWriter, get_logger
from .model import bn_keys, build_model
from .train import evaluate_cases, make_loader, set_seed, train_epochs

State = dict[str, torch.Tensor]


@dataclass(frozen=True)
class Method:
    name: str
    aggregate: bool
    keep_bn_local: bool
    pooled: bool


METHODS: dict[str, Method] = {
    "centralized": Method("centralized", aggregate=False, keep_bn_local=False, pooled=True),
    "local":       Method("local",       aggregate=False, keep_bn_local=False, pooled=False),
    "fedavg":      Method("fedavg",      aggregate=True,  keep_bn_local=False, pooled=False),
    "fedbn":       Method("fedbn",       aggregate=True,  keep_bn_local=True,  pooled=False),
}


# --------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------

def cpu_state(model: torch.nn.Module) -> State:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def weighted_average(states: list[State], weights: list[float],
                     skip: set[str] | None = None) -> State:
    """FedAvg's weighted mean over client states, for every key NOT in `skip`.

    Integer buffers (notably BatchNorm's `num_batches_tracked`, an int64) cannot be averaged --
    a weighted mean would produce a float and `load_state_dict` would reject or silently corrupt
    it. Those are copied from the highest-weighted client instead.
    """
    skip = skip or set()
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("aggregation weights must sum to > 0")
    heaviest = max(range(len(weights)), key=lambda i: weights[i])

    out: State = {}
    for k in states[0]:
        if k in skip:
            continue
        if not states[0][k].is_floating_point():
            out[k] = states[heaviest][k].clone()
            continue
        acc = torch.zeros_like(states[0][k], dtype=torch.float64)
        for s, w in zip(states, weights):
            acc += s[k].to(torch.float64) * w
        out[k] = (acc / total).to(states[0][k].dtype)
    return out


# --------------------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------------------

def run(cfg: Config, method_name: str) -> Path:
    """Run one experiment end to end. Returns the run directory."""
    if method_name not in METHODS:
        raise ValueError(f"unknown method {method_name!r}; pick from {sorted(METHODS)}")
    method = METHODS[method_name]

    run_dir = Path(cfg.paths.runs) / cfg.run_id(method_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("fedbrats", run_dir / "run.log")
    metrics = MetricsWriter(run_dir / "metrics.jsonl")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    index = load_index(cfg)
    hospitals = cfg.hospital_ids()

    train_cap = min(x for x in (cfg.train_per_hospital, cfg.max_train_cases) if x) \
        if (cfg.train_per_hospital or cfg.max_train_cases) else None
    train_cases = {h: select_cases(index, h, "train", train_cap) for h in hospitals}
    test_cases = {h: select_cases(index, h, "test", cfg.max_test_cases) for h in hospitals}

    with (run_dir / "config.json").open("w") as f:
        json.dump({k: str(v) for k, v in vars(cfg).items()}, f, indent=2, sort_keys=True)

    log.info(f"run_id={cfg.run_id(method_name)} method={method_name} dim={cfg.dim} device={device}")
    log.info(f"R={cfg.rounds} E={cfg.local_epochs} -> {cfg.total_epochs} total local epochs/hospital")
    for h in hospitals:
        log.info(f"  {h}: {len(train_cases[h])} train / {len(test_cases[h])} test cases")

    # --- clients: 'pooled' collapses the four hospitals into one client -------------------
    if method.pooled:
        pooled = sorted(c for h in hospitals for c in train_cases[h])
        clients = {"global": pooled}
        log.info(f"  pooled: {len(pooled)} train cases (centralized ceiling)")
    else:
        clients = dict(train_cases)

    loaders = {c: make_loader(cfg, ids, index) for c, ids in clients.items()}
    n_units = {c: len(loaders[c].dataset) for c in clients}

    # --- state ----------------------------------------------------------------------------
    model = build_model(cfg)
    init = cpu_state(model)
    bnk = bn_keys(model) if method.keep_bn_local else set()
    if method.keep_bn_local:
        log.info(f"  FedBN: keeping {len(bnk)} BatchNorm keys local")

    global_w: State = copy.deepcopy(init)
    init_bn: State = {k: init[k].clone() for k in bnk}
    bn_state: dict[str, State] = {}                       # fedbn: per-hospital BN
    own_w: dict[str, State] = {c: copy.deepcopy(init) for c in clients}   # local/centralized

    def start_state(client: str) -> State:
        if not method.aggregate:
            return own_w[client]
        return {**global_w, **bn_state.get(client, init_bn)}

    def eval_state(model_hospital: str) -> State:
        if not method.aggregate:
            return own_w["global" if method.pooled else model_hospital]
        return {**global_w, **bn_state.get(model_hospital, init_bn)}

    # --- rounds ---------------------------------------------------------------------------
    for rnd in range(1, cfg.rounds + 1):
        updates: dict[str, State] = {}
        for ci, (client, loader) in enumerate(loaders.items()):
            set_seed(cfg.seed + rnd * 1000 + ci)          # reproducible, distinct per client/round
            model.load_state_dict(start_state(client))
            loss = train_epochs(model, loader, cfg.local_epochs, cfg, device)
            updates[client] = cpu_state(model)
            log.info(f"round {rnd:>3}  {client:>7}  train_loss={loss:.4f}")

        if method.aggregate:
            names = list(updates)
            avg = weighted_average([updates[c] for c in names], [n_units[c] for c in names], skip=bnk)
            global_w.update(avg)                          # BN slots survive untouched under FedBN
            if method.keep_bn_local:
                for c in names:
                    bn_state[c] = {k: updates[c][k].clone() for k in bnk}
        else:
            own_w = updates                               # each client keeps its own full model

        # --- evaluate the federated model, before any further local training --------------
        for h in hospitals:
            mh = "global" if method.pooled or method.name == "fedavg" else h
            model.load_state_dict(eval_state(h))
            dice, _ = evaluate_cases(model, cfg, test_cases[h], device)
            metrics.write(run_id=cfg.run_id(method_name), method=method_name, dim=cfg.dim,
                          round=rnd, model_hospital=mh, test_hospital=h, split="test",
                          dice_wt=dice["wt"], dice_tc=dice["tc"], dice_et=dice["et"])
            log.info(f"round {rnd:>3}  eval {mh:>7} -> {h}  "
                     f"WT={dice['wt']:.4f} TC={dice['tc']:.4f} ET={dice['et']:.4f}")

    # --- final: 4x4 cross-hospital matrix for local-only ----------------------------------
    # Off-diagonal cells show H4's model collapsing on H1-H3 -- direct evidence the synthetic
    # shift creates a real domain gap. Run once at the end, not every round (16x eval cost).
    if method.name == "local":
        log.info("final cross-hospital matrix (local-only)")
        for mh in hospitals:
            model.load_state_dict(own_w[mh])
            for th in hospitals:
                if mh == th:
                    continue                              # diagonal already logged this round
                dice, _ = evaluate_cases(model, cfg, test_cases[th], device)
                metrics.write(run_id=cfg.run_id(method_name), method=method_name, dim=cfg.dim,
                              round=cfg.rounds, model_hospital=mh, test_hospital=th, split="test",
                              dice_wt=dice["wt"], dice_tc=dice["tc"], dice_et=dice["et"])
                log.info(f"  cross {mh} -> {th}  WT={dice['wt']:.4f}")

    # --- checkpoints ----------------------------------------------------------------------
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir(exist_ok=True)
    if method.aggregate:
        torch.save({"global": global_w, "bn": bn_state}, ckpt / "final.pt")
    else:
        torch.save(own_w, ckpt / "final.pt")
    log.info(f"done -> {run_dir}")
    return run_dir
