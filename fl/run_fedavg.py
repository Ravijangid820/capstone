"""Define and run a federated job in the FLARE Simulator.

The server always runs plain FedAvg; the *method* is selected by client-side
personalization (+ optional FedProx). One script covers our whole FL matrix:

    # FedAvg baseline (even split into 6 hospitals)
    uv run python fl/run_fedavg.py --data-root data/BraTS2021_Training_Data \
        --n-clients 6 --rounds 50 --epochs 1 --method fedavg --gpu 0

    # FedProx baseline
    ... --method fedprox --prox-mu 0.01 --gpu 0

    # FedBN (our primary personalization method)
    ... --method fedbn --personalization fedbn --gpu 0

    # Real FeTS institutional hospitals (n_clients inferred from the CSV)
    ... --fets-csv data/fets_partitioning.csv --method fedbn --personalization fedbn --gpu 0

CPU smoke (no GPU, tiny): drop --gpu, add --max-cases 6 --rounds 1.
Paths are made absolute before handing to clients (their CWD is the workspace).
"""

from __future__ import annotations

import argparse
import os

from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.model import PTModel
from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import ScriptRunner

from braintumor_fl.model import BratsUNet
from braintumor_fl.partition import get_partitions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--n-clients", type=int, default=6)
    p.add_argument("--fets-csv", default="", help="use real FeTS split; overrides --n-clients")
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--epochs", type=int, default=1, help="local epochs per round")
    p.add_argument("--method", default="fedavg", help="result tag: fedavg/fedprox/fedbn/personal_head")
    p.add_argument("--personalization", choices=["fedavg", "fedbn", "personal_head"], default="fedavg")
    p.add_argument("--synthetic-shift", action="store_true",
                   help="apply deterministic per-hospital scanner shift (synthetic non-IID)")
    p.add_argument("--prox-mu", type=float, default=0.0)
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers per client")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--workspace", default="data/fl_workspace")
    p.add_argument("--gpu", default=None, help="e.g. '0'; omit for CPU")
    p.add_argument("--threads", type=int, default=1, help="concurrent clients; 1 = 4GB-safe")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = os.path.abspath(args.data_root)
    results_dir = os.path.abspath(args.results_dir)
    fets_csv = os.path.abspath(args.fets_csv) if args.fets_csv else None

    # Number of hospitals: from the FeTS CSV if given, else the even-split count.
    n_clients = len(get_partitions(data_root, fets_csv=fets_csv)) if fets_csv else args.n_clients
    print(f"[job] method={args.method} | clients={n_clients} | rounds={args.rounds} | "
          f"personalization={args.personalization} | prox_mu={args.prox_mu}")

    job = FedJob(name=f"brats_{args.method}")
    job.to_server(FedAvg(num_clients=n_clients, num_rounds=args.rounds))
    job.to_server(PTModel(BratsUNet()))

    src = "--fets-csv " + fets_csv if fets_csv else f"--n-clients {n_clients}"
    shift_flag = " --synthetic-shift" if args.synthetic_shift else ""
    for i in range(n_clients):
        runner = ScriptRunner(
            script="fl/brats_client.py",
            script_args=(
                f"--data-root {data_root} --results-dir {results_dir} --method {args.method} "
                f"{src} --client-index {i} --personalization {args.personalization} "
                f"--prox-mu {args.prox_mu} --epochs {args.epochs} --max-cases {args.max_cases} "
                f"--batch-size {args.batch_size} --workers {args.workers}{shift_flag}"
            ),
        )
        job.to(runner, f"site-{i + 1}")

    job.simulator_run(os.path.abspath(args.workspace), gpu=args.gpu, threads=args.threads)


if __name__ == "__main__":
    main()
