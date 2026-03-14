local _M = {}

-- user settings
_M.python_executable = "/home/martin/Development/xournalpp_htr/.venv/bin/python"
_M.xournalpp_htr_path = "/home/martin/Development/xournalpp_htr/xournalpp_htr/run_htr.py"
_M.pipeline = "dummy"
_M.output_file = "/home/martin/Development/xournalpp_htr/tests/test_1_from_Xpp.pdf"
_M.debug_HTR_command = false
-- TODO: allow UI to set other parameters as well of `xournalpp_htr`.

-- TODO replace later w/ temp exported file - filename will be derived automatically
_M.filename = "/home/martin/Development/xournalpp_htr/tests/test_1.xoj"

return _M