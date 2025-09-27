"""Script to perform HTR on Xournal(++) document."""

from xournalpp_htr.pipeline import export_xournalpp_to_pdf_with_htr
from xournalpp_htr.utils import parse_arguments

# Step 4: Next steps
#
# I want to build prediction code that can run both in a CLI and in a notebook like this here. Also, I'd like to be able to set the model flexibly.


if __name__ == "__main__":
    args = parse_arguments()
    export_xournalpp_to_pdf_with_htr(args)
