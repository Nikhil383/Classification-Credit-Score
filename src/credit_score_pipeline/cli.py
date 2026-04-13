from __future__ import annotations

import argparse
import json

from .predict import predict_from_csv
from .train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit score end-to-end pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train model and persist artifacts")
    train_parser.add_argument("--pretty", action="store_true", help="Pretty-print metrics")

    predict_parser = subparsers.add_parser("predict", help="Generate predictions from CSV")
    predict_parser.add_argument("--input", required=True, help="Input CSV path")
    predict_parser.add_argument("--output", required=True, help="Output CSV path")

    args = parser.parse_args()

    if args.command == "train":
        metrics = run_training()
        if args.pretty:
            print(json.dumps(metrics, indent=2))
        else:
            print(metrics)

    if args.command == "predict":
        predict_from_csv(input_csv=args.input, output_csv=args.output)
        print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()
