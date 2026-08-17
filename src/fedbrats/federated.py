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
from .data import load_index, select_cases, train_val_cases
from .logging_utils import MetricsWriter, get_logger, guard_run_dir
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

def run(cfg: Config, method_name: str, overwrite: bool = False) -> Path:
    """Run one experiment end to end. Returns the run directory."""
    if method_name not in METHODS:
        raise ValueError(f"unknown method {method_name!r}; pick from {sorted(METHODS)}")
    method = METHODS[method_name]

    run_dir = cfg.run_dir(method_name)
    guard_run_dir(run_dir, overwrite)                     # never append into an earlier experiment
    run_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("fedbrats", run_dir / "run.log", mode="w")
    metrics = MetricsWriter(run_dir / "metrics.jsonl")
    per_case_log = MetricsWriter(run_dir / "per_case.jsonl")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    index = load_index(cfg)
    hospitals = cfg.hospital_ids()

    train_cap = min(x for x in (cfg.train_per_hospital, cfg.max_train_cases) if x) \
        if (cfg.train_per_hospital or cfg.max_train_cases) else None
    split = {h: train_val_cases(cfg, index, h, train_cap) for h in hospitals}
    train_cases = {h: split[h][0] for h in hospitals}
    val_cases = {h: split[h][1] for h in hospitals}
    test_cases = {h: select_cases(index, h, "test", cfg.max_test_cases) for h in hospitals}

    with (run_dir / "config.json").open("w") as f:
        json.dump(cfg.to_dict(), f, indent=2, sort_keys=True, default=str)

    log.info(f"run_id={cfg.run_id(method_name)} method={method_name} dim={cfg.dim} device={device}")
    if cfg.tag:
        log.info(f"tag={cfg.tag} -> {run_dir}")
    log.info(f"R={cfg.rounds} E={cfg.local_epochs} -> {cfg.total_epochs} total local epochs/hospital")
    log.info(f"lr={cfg.lr} schedule={cfg.lr_schedule} augment={cfg.augment} tta={cfg.tta} "
             f"postproc_min_voxels={cfg.postproc_min_voxels} select_by={cfg.select_by}")
    for h in hospitals:
        log.info(f"  {h}: {len(train_cases[h])} train / {len(val_cases[h])} val / "
                 f"{len(test_cases[h])} test cases")

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

    def snapshot() -> dict:
        """Everything needed to reconstruct this round's models later, for best-val selection."""
        return {"global": copy.deepcopy(global_w), "bn": copy.deepcopy(bn_state),
                "own": copy.deepcopy(own_w)}

    def restore(snap: dict) -> None:
        nonlocal global_w, bn_state, own_w
        global_w, bn_state, own_w = snap["global"], snap["bn"], snap["own"]

    def score(hospital: str, cases: list[str], split_name: str, rnd: int,
              stage: str, tta: bool) -> dict[str, float]:
        """Evaluate one hospital's serving model and log both the mean and every case."""
        mh = "global" if method.pooled or method.name == "fedavg" else hospital
        model.load_state_dict(eval_state(hospital))
        dice, per_case = evaluate_cases(model, cfg, cases, device, tta=tta)
        common = dict(run_id=cfg.run_id(method_name), method=method_name, dim=cfg.dim, round=rnd,
                      stage=stage, model_hospital=mh, test_hospital=hospital, split=split_name)
        metrics.write(**common, tta=tta,
                      dice_wt=dice["wt"], dice_tc=dice["tc"], dice_et=dice["et"])
        per_case_log.write_many([{**common, "case_id": c["case_id"], "dice_wt": c["wt"],
                                  "dice_tc": c["tc"], "dice_et": c["et"]} for c in per_case])
        return dice

    # --- rounds ---------------------------------------------------------------------------
    best = {"round": 0, "val_wt": float("-inf"), "snap": None}

    for rnd in range(1, cfg.rounds + 1):
        lr = cfg.round_lr(rnd)
        updates: dict[str, State] = {}
        for ci, (client, loader) in enumerate(loaders.items()):
            set_seed(cfg.seed + rnd * 1000 + ci)          # reproducible, distinct per client/round
            model.load_state_dict(start_state(client))
            loss = train_epochs(model, loader, cfg.local_epochs, cfg, device, lr=lr)
            updates[client] = cpu_state(model)
            log.info(f"round {rnd:>3}  {client:>7}  train_loss={loss:.4f}  lr={lr:.2e}")

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
        # No TTA here: the learning curve costs 4x with it, and the curve is used for shape,
        # not for the headline number. The reported model is re-scored with TTA below.
        if cfg.scores_test(rnd):
            for h in hospitals:
                dice = score(h, test_cases[h], "test", rnd, stage="round", tta=False)
                log.info(f"round {rnd:>3}  eval {h:>3}  "
                         f"WT={dice['wt']:.4f} TC={dice['tc']:.4f} ET={dice['et']:.4f}")
        else:
            log.info(f"round {rnd:>3}  test eval skipped (eval_test_every={cfg.eval_test_every})")

        # --- validation: the only signal model selection is allowed to read ----------------
        if cfg.val_per_hospital > 0:
            vals = [score(h, val_cases[h], "val", rnd, stage="round", tta=False)["wt"]
                    for h in hospitals]
            val_wt = sum(vals) / len(vals)
            log.info(f"round {rnd:>3}  VAL mean WT={val_wt:.4f}"
                     f"{'  <- best so far' if val_wt > best['val_wt'] else ''}")
            if val_wt > best["val_wt"]:
                best = {"round": rnd, "val_wt": val_wt, "snap": snapshot()}

    # --- select the model to report -------------------------------------------------------
    # "last" keeps the baseline's behaviour. "best_val" rewinds to the round that scored best on
    # validation -- never on test, so the reported figure stays a measurement of a held-out set
    # rather than the maximum of 25 draws from it.
    reported_round = cfg.rounds
    if cfg.select_by == "best_val" and best["snap"] is not None:
        restore(best["snap"])
        reported_round = best["round"]
        log.info(f"selected round {reported_round} by validation WT={best['val_wt']:.4f} "
                 f"(of {cfg.rounds} rounds)")

    # --- final: re-score the selected model, this time with TTA ---------------------------
    # `stage="final"` marks these rows; the analysis reads them in preference to the round rows,
    # so the headline number and the learning curve can differ in inference cost without either
    # being ambiguous about which one it is.
    final_tta = cfg.tta
    log.info(f"final evaluation at round {reported_round} (tta={final_tta})")
    for h in hospitals:
        dice = score(h, test_cases[h], "test", reported_round, stage="final", tta=final_tta)
        log.info(f"  final {h}  WT={dice['wt']:.4f} TC={dice['tc']:.4f} ET={dice['et']:.4f}")

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
                dice, _ = evaluate_cases(model, cfg, test_cases[th], device, tta=final_tta)
                metrics.write(run_id=cfg.run_id(method_name), method=method_name, dim=cfg.dim,
                              round=reported_round, stage="cross", model_hospital=mh,
                              test_hospital=th, split="test", tta=final_tta,
                              dice_wt=dice["wt"], dice_tc=dice["tc"], dice_et=dice["et"])
                log.info(f"  cross {mh} -> {th}  WT={dice['wt']:.4f}")

    # --- checkpoints ----------------------------------------------------------------------
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir(exist_ok=True)
    payload = {"global": global_w, "bn": bn_state} if method.aggregate else own_w
    torch.save({"state": payload, "round": reported_round, "select_by": cfg.select_by,
                "config": cfg.to_dict()}, ckpt / "final.pt")

    summary = {"run_id": cfg.run_id(method_name), "method": method_name, "dim": cfg.dim,
               "tag": cfg.tag, "rounds": cfg.rounds, "reported_round": reported_round,
               "select_by": cfg.select_by, "best_val_wt": best["val_wt"] if best["snap"] else None,
               "tta": final_tta}
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    log.info(f"done -> {run_dir}")
    return run_dir
