from pathlib import Path
import subprocess


def export_to_pdf_with_xournalpp(input_file: Path, output_file: Path) -> None:
    """Export a Xournal(++) file to PDF using Xournal++.

    This function uses the `xournalpp` command-line tool to convert a Xournal(++) file
    specified by `input_file` to a PDF file specified by `output_file`. If the export fails
    for any reason, an exception is raised.

    :param input_file: Path to the input Xournal(++) file.
    :type input_file: Path
    :param output_file: Path to the output PDF file.
    :type output_file: Path
    :raises RuntimeError: If the PDF export fails.

    .. note::
       Ensure that Xournal++ is installed and available in the system's PATH.

    .. code-block:: python

        from pathlib import Path
        from xournalpp_htr.utils import export_to_pdf_with_xournalpp

        input_path = Path('/path/to/input/file.xopp')
        output_path = Path('/path/to/output/file.pdf')

        try:
            export_to_pdf_with_xournalpp(input_path, output_path)
            print("Export successful!")
        except RuntimeError as e:
            print(f"Export failed: {e}")
    """

    if not input_file.exists():
        raise ValueError(f'input file "{input_file}" does not exist.')

    command = f'xournalpp {input_file} -p {output_file}'
    export_result = subprocess.run(command,
                                   shell=True,
                                   capture_output=True)

    return_code_fail = export_result.returncode != 0
    stdout_fail = 'PDF file successfully created' not in export_result.stderr.decode('utf-8')
    file_existing_fail = not output_file.exists()

    if return_code_fail or stdout_fail or file_existing_fail:
        raise RuntimeError('PDF export failed')

    return output_file