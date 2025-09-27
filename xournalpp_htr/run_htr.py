"""Script to perform HTR on Xournal(++) document."""

from xournalpp_htr.pipeline import export_xournalpp_to_pdf_with_htr
from xournalpp_htr.utils import parse_arguments

if __name__ == "__main__":
    args = parse_arguments()
    export_xournalpp_to_pdf_with_htr(args)
