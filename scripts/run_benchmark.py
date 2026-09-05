"""Script to benchmark HTR pipeline against the xournalpp_htr_benchmark dataset."""

import argparse
import json
from pathlib import Path

from xournalpp_htr.benchmark import run_benchmark, write_html_report


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
    parser.add_argument(
        "-o",
        "--html-report",
        type=Path,
        required=False,
        default=None,
        help=(
            "Write a self-contained HTML report with page renders, prediction "
            "overlays and per-word analysis to this file. Passing it enables the "
            "extra analysis, which is slower; omitting it keeps the plain run."
        ),
    )
    return vars(parser.parse_args(cli_string.split() if cli_string else None))


if __name__ == "__main__":
    args = parse_arguments()
    result = run_benchmark(
        args["pipeline"], collect_details=args["html_report"] is not None
    )

    if args["html_report"] is not None:
        write_html_report(result, args["html_report"])

    if args["format"] == "json":
        print(
            json.dumps(
                {
                    "pipeline": args["pipeline"],
                    "precision": result.precision,
                    "recall": result.recall,
                    "cer": result.cer,
                    "cer_case_insensitive": result.cer_case_insensitive,
                    "recall_times_accuracy": result.recall_times_accuracy,
                    "word_accuracy": result.word_accuracy,
                    "n_gt_words": result.n_gt_words,
                    "n_predicted_words": result.n_predicted_words,
                    "n_matched": result.n_matched,
                    "html_report": str(args["html_report"])
                    if args["html_report"]
                    else None,
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
        print(f"R×(1-CER): {result.recall_times_accuracy:.1%}  (recall × (1 − CER_ci))")
        print(
            f"Word Acc : {result.word_accuracy:.1%}  ({round(result.word_accuracy * result.n_matched)}/{result.n_matched} matched words)"
        )
        if args["html_report"] is not None:
            print(f"Report   : {args['html_report']}")
