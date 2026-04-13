import argparse

from credit_score_pipeline.predict import predict_from_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    predict_from_csv(args.input, args.output)
    print(f"Saved predictions to {args.output}")
