local _M = {}

-- user settings
_M.python_executable = "/path/to/python"
_M.xournalpp_htr_path = "/path/to/xournalpp_htr/run_htr.py"
_M.pipeline = "dummy"
_M.output_file = "/path/to/output.pdf"
_M.debug_HTR_command = false
-- TODO: allow UI to set other parameters as well of `xournalpp_htr`.

-- TODO replace later w/ temp exported file - filename will be derived automatically
_M.filename = "/path/to/input.xoj"

return _M