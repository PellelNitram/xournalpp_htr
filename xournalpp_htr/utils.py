from pathlib import Path
import subprocess


def export_to_pdf_with_xournalpp(input_file: Path, output_file: Path) -> None:
    """Export Xournal(++) file to PDF using Xournal++.
    
    TODO
    """
    # TODO: Needs to raise Exception if export fails

    export_result = subprocess.run(f'xournalpp {input_file} -p {output_file}',
                               shell=True,
                               capture_output=True)

    return_code_fail = export_result.returncode != 0
    stdout_fail = 'PDF file successfully created' not in export_result.stderr.decode('utf-8')
    file_existing_fail = not output_file.exists()

    if return_code_fail or stdout_fail or file_existing_fail:
        raise RuntimeError('PDF export failed')

    return output_file