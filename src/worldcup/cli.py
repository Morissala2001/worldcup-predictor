"""Command-line interface: `worldcup download | train | predict | simulate`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import data as data_module
from . import model as model_module
from . import tournament
from .display import label, percent_split

DEFAULT_BRACKET = Path("data/bracket_2030.json")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=data_module.DEFAULT_PATH, help="path to results.csv")
    parser.add_argument("--start", default="1994-01-01", help="ignore matches before this date")
    parser.add_argument("--window", type=int, default=10, help="number of past matches used to measure form")


def _load(args):
    matches = data_module.modern_era(data_module.load_results(args.data), start=args.start)
    return matches, model_module.train(matches, window=args.window)


def _print_metrics(model: model_module.MatchModel) -> None:
    m = model.metrics
    print(f"Test set ({m['split']} split): {m['n_test']} matches, {m['test_from']} -> {m['test_to']}")
    print(f"  baseline (always bet on the home side): {m['baseline']:.1%}")
    print(f"  model accuracy                         : {m['accuracy']:.1%}")
    print("Feature importances:")
    for name, value in model.importances.sort_values(ascending=False).items():
        print(f"  {name:<26} {value:.3f}")


def cmd_download(args) -> None:
    path = data_module.download_results(args.data)
    print(f"Saved the latest results to {path}")


def cmd_train(args) -> None:
    matches = data_module.modern_era(data_module.load_results(args.data), start=args.start)
    model = model_module.train(matches, window=args.window, split=args.split)
    print(f"{len(matches)} matches since {args.start}; {model.metrics['n_train']} used for training")
    _print_metrics(model)


def cmd_predict(args) -> None:
    matches, model = _load(args)
    prediction = model_module.predict_match(
        model, matches, args.team_a, args.team_b, neutral=not args.home, friendly=args.friendly
    )
    venue = f"at {args.team_a}" if args.home else "on neutral ground"
    print(f"{label(args.team_a)} vs {label(args.team_b)} ({venue})")
    percent_a, percent_b = percent_split(prediction.prob_a)
    print(f"  {args.team_a}: {percent_a}%   {args.team_b}: {percent_b}%")
    print(f"  -> {label(prediction.winner)} ({prediction.win_probability:.0%})")


def cmd_simulate(args) -> None:
    matches, model = _load(args)
    bracket = tournament.load_bracket(args.bracket)
    forms = model_module.team_forms(matches, tournament.bracket_teams(bracket), model.window)
    probs = model_module.pairwise_win_probabilities(model, forms)
    win_prob = lambda a, b: probs[(a, b)]  # noqa: E731

    rounds = tournament.simulate(bracket, win_prob)
    for results in rounds:
        print(f"\n== {tournament.round_name(len(results)).upper()} ==")
        for r in results:
            print(f"{label(r.team_a):<28} vs {label(r.team_b):<28} -> {label(r.winner)} ({r.probability:.0%})")
    print(f"\nChampion (favourite always wins): {label(tournament.champion(rounds))}")

    if args.runs:
        odds = tournament.monte_carlo(bracket, win_prob, runs=args.runs, seed=args.seed)
        ranking = sorted(odds, key=lambda t: odds[t].get(tournament.CHAMPION, 0), reverse=True)
        print(f"\nTitle odds over {args.runs} simulated tournaments (each match drawn at random):")
        for team in ranking[: args.top]:
            print(f"  {label(team):<28} {odds[team].get(tournament.CHAMPION, 0):6.1%}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="worldcup", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch the latest results.csv")
    p.add_argument("--data", type=Path, default=data_module.DEFAULT_PATH)
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("train", help="train the model and print its accuracy")
    _add_common(p)
    p.add_argument("--split", choices=["temporal", "random"], default="temporal")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="predict one match (team names are in English)")
    _add_common(p)
    p.add_argument("team_a")
    p.add_argument("team_b")
    p.add_argument("--home", action="store_true", help="team_a plays at home (default: neutral ground)")
    p.add_argument("--friendly", action="store_true", help="the match is a friendly")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("simulate", help="simulate a knockout bracket")
    _add_common(p)
    p.add_argument("--bracket", type=Path, default=DEFAULT_BRACKET, help="JSON list of [team_a, team_b] pairs")
    p.add_argument("--runs", type=int, default=0, help="also run this many Monte Carlo tournaments")
    p.add_argument("--top", type=int, default=10, help="how many teams to show in the Monte Carlo table")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_simulate)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # flag emoji on Windows consoles
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
