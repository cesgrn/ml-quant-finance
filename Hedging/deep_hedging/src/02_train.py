import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from utils.market import simulate_gbm
from utils.models import FeedForwardHedger, NoTransactionBandNet
from utils.strategies import bs_delta, whalley_wilmott
from utils.training import evaluate, train
from utils.style import setup, style

if __name__ == "__main__":
    os.makedirs("data/results", exist_ok=True)
    cost = cfg.COST

    # 1. Train both networks; each starts from the same seeded initial weights
    builders = {"Feed-forward": FeedForwardHedger,
                "No-transaction band": NoTransactionBandNet}
    models, histories = {}, {}
    for name, build in builders.items():
        torch.manual_seed(cfg.TRAIN_SEED)          # fixes the initial weights
        model = build(cfg.T)                       # weights are drawn here
        print(f"\nTraining {name} (cost {cost:.2%}, lambda {cfg.RISK_AVERSION})")
        histories[name] = train(model, cfg, cost, seed=cfg.TRAIN_SEED)
        torch.save(model.state_dict(),
                   f"data/results/{name.replace(' ', '_').lower()}_{cost}.pt")
        models[name] = model

    # 2. Evaluate everything on the same 20,000 test paths
    test_paths = simulate_gbm(cfg.N_TEST_PATHS, cfg.S0, cfg.R, cfg.SIGMA, cfg.T,
                              cfg.N_STEPS, seed=cfg.TEST_SEED)
    strategies = {"BS delta": bs_delta,
                  "Whalley-Wilmott": whalley_wilmott(cost, cfg.RISK_AVERSION, cfg.R),
                  **models}
    results = pd.DataFrame([{"strategy": n, **evaluate(s, test_paths, cfg, cost)}
                            for n, s in strategies.items()]).set_index("strategy")
    print("\n", results[["mean_pnl", "std_pnl", "cvar_95", "entropic_risk",
                         "price", "turnover"]].round(3))
    results.to_csv(f"data/results/comparison_{cost}.csv")

    # 3. Learning curves: the paper's key figure
    setup()
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, h in histories.items():
        ax.plot(h, label=name, alpha=0.85, **style(name, lw=1.2))   # noisy: thinner
    ax.axhline(results.loc["Whalley-Wilmott", "entropic_risk"],
               label="Whalley-Wilmott", **style("Whalley-Wilmott"))
    ax.axhline(results.loc["BS delta", "entropic_risk"],
               label="BS delta", **style("BS delta"))
    ax.set_ylim(top=results.loc["BS delta", "entropic_risk"] * 1.5)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Entropic risk (training batch)")
    ax.set_title(f"Learning curves, cost = {cost:.0%}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"data/results/learning_curves_{cost}.png", dpi=150)