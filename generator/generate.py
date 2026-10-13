"""Generate synthetic open banking data for the credit-scoring case study.

Writes three JSONL files to --out:
  accounts.jsonl      one document per bank account, transactions nested inside
                      (shaped like an open banking API response)
  applications.jsonl  one loan application per user
  outcomes.jsonl      whether the loan defaulted within 12 months (skip with --no-outcomes)

Examples:
  python generator/generate.py --batch train --users 3000 --start 2024-01-01 --end 2025-09-30
  python generator/generate.py --batch prod --users 800 --last-days 30 --no-outcomes
  python generator/generate.py --batch prod --users 800 --last-days 30 --no-outcomes --drift
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np

HISTORY_DAYS = 210          # bank history fetched at application time (features use 180 days)
OVERDRAFT_LIMIT = -400.0    # direct debits are rejected below this balance
REJECTION_FEE = 20.0
TARGET_DEFAULT_RATE = 0.10


@dataclass
class Persona:
    name: str
    weight: float
    drift_weight: float
    income: tuple[float, float]       # mean and std of monthly income (EUR)
    rent_ratio: tuple[float, float]   # rent as a share of income (uniform range)
    spend_ratio: tuple[float, float]  # total monthly spending / income (uniform range)
    gambler_prob: float
    hidden_risk: float                # risk the data cannot fully show


PERSONAS = [
    Persona("salaried", 0.55, 0.40, (2600, 600), (0.20, 0.35), (0.75, 0.95), 0.08, 0.0),
    Persona("freelancer", 0.15, 0.15, (2800, 900), (0.20, 0.40), (0.80, 1.00), 0.10, 0.3),
    Persona("student", 0.12, 0.10, (900, 150), (0.30, 0.50), (0.90, 1.05), 0.12, 0.4),
    Persona("fragile", 0.18, 0.35, (1500, 250), (0.35, 0.55), (0.95, 1.15), 0.30, 0.8),
]
START_BALANCE = {"salaried": (300, 3000), "freelancer": (200, 4000), "student": (0, 600), "fragile": (-150, 400)}
EMPLOYERS = ["ACME SAS", "NOVA TECH", "BOULANGERIE MARTIN", "LOGISTIQUE OUEST", "CLINIQUE DU PARC"]
LANDLORDS = ["SCI DUPONT", "FONCIA GESTION", "M LEROY", "HABITAT 35"]
DISCRETIONARY = {  # category: (description options, share of budget, tx per month range)
    "groceries": (["CB SUPERMARCHE", "CB EPICERIE DU COIN", "CB BOULANGERIE"], 0.40, (8, 12)),
    "restaurants": (["CB RESTAURANT", "CB BRASSERIE DU PORT", "CB PIZZERIA"], 0.15, (3, 6)),
    "transport": (["CB CARBURANT STATION", "CB TRANSPORT METRO", "CB PEAGE AUTOROUTE"], 0.15, (3, 6)),
    "shopping": (["CB BOUTIQUE EN LIGNE", "CB MAGASIN VETEMENTS"], 0.20, (2, 4)),
    "cash": (["RETRAIT DAB"], 0.10, (1, 2)),
}


def at(day: date, rng: np.random.Generator) -> datetime:
    return datetime.combine(day, time(int(rng.integers(7, 22)), int(rng.integers(0, 60))))


def month_starts(start: date, end: date):
    current = date(start.year, start.month, 1)
    while current <= end:
        yield current
        current = date(current.year + (current.month == 12), current.month % 12 + 1, 1)


def simulate_user(i: int, batch: str, persona: Persona, app_date: date, drift: bool, rng: np.random.Generator):
    user_id, account_id, app_id = f"usr_{batch}_{i:05d}", f"acc_{batch}_{i:05d}", f"app_{batch}_{i:05d}"
    income = max(rng.normal(*persona.income), 400.0)
    rent = income * rng.uniform(*persona.rent_ratio) * (1.15 if drift else 1.0)
    monthly_spend = income * rng.uniform(*persona.spend_ratio)
    gambler = rng.random() < persona.gambler_prob * (1.8 if drift else 1.0)
    gambling_ratio = rng.uniform(0.05, 0.30) if gambler else 0.0

    history_start = app_date - timedelta(days=HISTORY_DAYS)
    fetched_at = datetime.combine(app_date, time(18, 0))
    events = []  # (when, amount, description, is_direct_debit)

    def add(day: int, month: date, amount: float, description: str, direct_debit: bool = False):
        when = at(date(month.year, month.month, day), rng)
        if datetime.combine(history_start, time.min) <= when < fetched_at:
            events.append((when, round(amount, 2), description, direct_debit))

    for month in month_starts(history_start, app_date):
        # Income
        if persona.name == "salaried":
            add(28, month, income * (1 + rng.normal(0, 0.02)), f"VIR SALAIRE {rng.choice(EMPLOYERS)}")
        elif persona.name == "freelancer":
            n_invoices = 0 if rng.random() < 0.15 else rng.poisson(1.6)
            if n_invoices:
                total = income * rng.lognormal(0, 0.45)
                for share in rng.dirichlet(np.ones(n_invoices)):
                    add(int(rng.integers(1, 29)), month, total * share, f"VIR CLIENT FACTURE F{rng.integers(1000, 9999)}")
        elif persona.name == "student":
            add(5, month, income * 0.6, "VIR SALAIRE JOB ETUDIANT")
            add(6, month, income * 0.4, "VIR CAF ALLOCATIONS")
        else:  # fragile: salary sometimes missing, plus benefits
            if rng.random() > 0.10:
                add(28, month, income * 0.85, f"VIR SALAIRE {rng.choice(EMPLOYERS)}")
            add(6, month, income * 0.15, "VIR CAF ALLOCATIONS")

        # Fixed charges (direct debits can be rejected)
        add(5, month, -rent, f"PRLV LOYER {rng.choice(LANDLORDS)}", True)
        electricity, internet, streaming = rng.uniform(45, 110), rng.uniform(25, 55), rng.uniform(9, 18)
        add(10, month, -electricity, "PRLV ELECTRICITE", True)
        add(15, month, -internet, "PRLV BOX INTERNET MOBILE", True)
        add(12, month, -streaming, "PRLV ABONNEMENT STREAMING", True)

        # Discretionary spending and gambling
        budget = max(monthly_spend - rent - electricity - internet - streaming, 150.0)
        gambling = budget * gambling_ratio
        for options, share, (low, high) in DISCRETIONARY.values():
            n = int(rng.integers(low, high + 1))
            for amount in (budget - gambling) * share * rng.dirichlet(np.ones(n)):
                add(int(rng.integers(1, 29)), month, -amount, str(rng.choice(options)))
        if gambling:
            n = int(rng.integers(2, 9))
            for amount in gambling * rng.dirichlet(np.ones(n)):
                add(int(rng.integers(1, 29)), month, -amount, str(rng.choice(["CB PARIS SPORTIF EN LIGNE", "CB CASINO EN LIGNE"])))
        if rng.random() < 0.3:  # transfers from friends, left uncategorized on purpose
            add(int(rng.integers(1, 29)), month, rng.uniform(20, 150), "VIR RECU DE PROCHE")
        surplus = income - monthly_spend
        if surplus > 100:
            add(2, month, -surplus * rng.uniform(0.3, 0.7), "VIR VERS LIVRET EPARGNE")

    # Replay events in time order to compute balances and rejected direct debits
    balance = round(rng.uniform(*START_BALANCE[persona.name]), 2)
    transactions = []
    for when, amount, description, direct_debit in sorted(events, key=lambda e: e[0]):
        if direct_debit and balance + amount < OVERDRAFT_LIMIT:
            when, amount, description = when + timedelta(minutes=1), -REJECTION_FEE, "FRAIS REJET PRELEVEMENT"
        elif description.startswith("VIR VERS LIVRET") and balance + amount < 0:
            continue  # nobody saves money while overdrawn
        balance = round(balance + amount, 2)
        transactions.append({
            "transaction_id": f"{account_id}-{len(transactions):05d}",
            "booked_at": when.isoformat(timespec="seconds"),
            "booking_date": when.date().isoformat(),
            "amount": amount,
            "description": description,
            "balance_after": balance,
        })

    # Risk signals as the lender would observe them over the following 12 months
    window_start = (app_date - timedelta(days=180)).isoformat()
    recent = [t for t in transactions if t["booking_date"] >= window_start]
    n_rejected = sum(t["description"] == "FRAIS REJET PRELEVEMENT" for t in recent)
    negative_days = len({t["booking_date"] for t in recent if t["balance_after"] < 0})
    risk_logit = (
        0.9 * persona.hidden_risk
        + 0.30 * min(n_rejected, 6)
        + 0.025 * min(negative_days, 90)
        + 3.0 * gambling_ratio
        + 2.5 * max(rent / income - 0.33, 0.0)
        + rng.normal(0, 1.2)
    )

    account = {
        "_id": account_id,
        "user_id": user_id,
        "bank": "BANQUE DEMO",
        "currency": "EUR",
        "fetched_at": fetched_at.isoformat(timespec="seconds"),
        "current_balance": balance,
        "transactions": transactions,
    }
    application = {
        "_id": app_id,
        "user_id": user_id,
        "account_id": account_id,
        "application_date": app_date.isoformat(),
        "requested_amount": int(rng.choice([1000, 2000, 3000, 5000, 8000])),
        "partner_id": str(rng.choice(["lender_a", "lender_b", "lender_c"])),
    }
    return account, application, risk_logit


def calibrated_intercept(logits: np.ndarray, target: float) -> float:
    """Find the intercept that gives the target default rate (bisection)."""
    low, high = -15.0, 15.0
    for _ in range(60):
        mid = (low + high) / 2
        rate = np.mean(1 / (1 + np.exp(-(mid + logits))))
        low, high = (mid, high) if rate < target else (low, mid)
    return (low + high) / 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", required=True, help="prefix for ids, e.g. train or prod")
    parser.add_argument("--users", type=int, default=3000)
    parser.add_argument("--start", type=date.fromisoformat, help="first application date")
    parser.add_argument("--end", type=date.fromisoformat, help="last application date")
    parser.add_argument("--last-days", type=int, help="use applications from the last N days instead of --start/--end")
    parser.add_argument("--drift", action="store_true", help="shift the population: more fragile users, higher rents, more gambling")
    parser.add_argument("--no-outcomes", action="store_true", help="recent applications have no known outcome yet")
    parser.add_argument("--out", type=Path, help="output folder (default: data/<batch>)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.last_days:
        args.end = date.today() - timedelta(days=1)
        args.start = args.end - timedelta(days=args.last_days - 1)
    if not (args.start and args.end):
        parser.error("give --start and --end, or --last-days")
    out = args.out or Path("data") / args.batch
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    weights = np.array([p.drift_weight if args.drift else p.weight for p in PERSONAS])
    span = (args.end - args.start).days + 1
    accounts, applications, logits = [], [], []
    for i in range(1, args.users + 1):
        persona = PERSONAS[rng.choice(len(PERSONAS), p=weights)]
        app_date = args.start + timedelta(days=int(rng.integers(0, span)))
        account, application, logit = simulate_user(i, args.batch, persona, app_date, args.drift, rng)
        accounts.append(account)
        applications.append(application)
        logits.append(logit)

    with open(out / "accounts.jsonl", "w") as f:
        f.writelines(json.dumps(a) + "\n" for a in accounts)
    with open(out / "applications.jsonl", "w") as f:
        f.writelines(json.dumps(a) + "\n" for a in applications)

    n_tx = sum(len(a["transactions"]) for a in accounts)
    print(f"{len(accounts)} accounts, {n_tx} transactions ({n_tx / len(accounts):.0f} per account) -> {out}")

    if not args.no_outcomes:
        logits = np.array(logits)
        p_default = 1 / (1 + np.exp(-(calibrated_intercept(logits, TARGET_DEFAULT_RATE) + logits)))
        defaulted = rng.random(len(p_default)) < p_default
        with open(out / "outcomes.jsonl", "w") as f:
            for application, flag in zip(applications, defaulted):
                outcome_date = date.fromisoformat(application["application_date"]) + timedelta(days=365)
                f.write(json.dumps({
                    "_id": application["_id"],
                    "application_id": application["_id"],
                    "defaulted": bool(flag),
                    "outcome_date": outcome_date.isoformat(),
                }) + "\n")
        print(f"default rate: {defaulted.mean():.1%}")


if __name__ == "__main__":
    main()
