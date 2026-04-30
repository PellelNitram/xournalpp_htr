"""Script to benchmark HTR pipeline against the xournalpp_htr_benchmark dataset."""

import argparse
import json

from xournalpp_htr.benchmark import run_benchmark


def parse_arguments(cli_string: None | str = None):
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(
        description="Benchmark an HTR pipeline against the xournalpp_htr_benchmark dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--pipeline",
        type=str,
        required=False,
        default="2024-07-18_htr_pipeline",
        help="The pipeline to benchmark.",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["human", "json"],
        default="human",
        help="Output format.",
    )
    return vars(parser.parse_args(cli_string.split() if cli_string else None))


if __name__ == "__main__":
    args = parse_arguments()
    result = run_benchmark(args["pipeline"])

    if args["format"] == "json":
        print(
            json.dumps(
                {
                    "pipeline": args["pipeline"],
                    "precision": result.precision,
                    "recall": result.recall,
                    "cer": result.cer,
                    "cer_case_insensitive": result.cer_case_insensitive,
                    "n_gt_words": result.n_gt_words,
                    "n_predicted_words": result.n_predicted_words,
                    "n_matched": result.n_matched,
                },
                indent=2,
            )
        )
    else:
        print(f"Pipeline : {args['pipeline']}")
        print(
            f"Precision: {result.precision:.1%}  ({result.n_matched}/{result.n_predicted_words} predictions matched)"
        )
        print(
            f"Recall   : {result.recall:.1%}  ({result.n_matched}/{result.n_gt_words} GT words matched)"
        )
        print(f"CER      : {result.cer:.1%}  (case-sensitive)")
        print(f"CER      : {result.cer_case_insensitive:.1%}  (case-insensitive)")
