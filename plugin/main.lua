function initUi()
  app.registerUi({["menu"] = "Xournal++ HTR", ["callback"] = "run", ["accelerator"] = "<Control>F1"});
end
  
function run()
  -- user settings
  python_executable = "/home/martin/anaconda3/envs/xournalpp_htr/bin/python"
  xournalpp_htr_path = "/home/martin/Development/xournalpp_htr/xournalpp_htr/xournalpp_htr.py"
  model = "dummy"
  output_file = "/home/martin/Development/xournalpp_htr/tests/test_1_from_Xpp.pdf"
  -- TODO: allow UI to set other parameters as well of `xournalpp_htr`.

  -- TODO replace later w/ temp exported file - filename will be derived automatically
  filename = "/home/martin/Development/xournalpp_htr/tests/test_1.xoj"

  local result = app.msgbox("Exports starts now, please wait until finished", {[1] = "Continue", [2] = "Cancel"})
  if result == 1 then
    command = python_executable .. " " .. xournalpp_htr_path .. " -if " .. filename .. " -m " .. model .. " -of " .. output_file
    os.execute(command)
    app.msgbox("Export finished!", {[1] = "Continue"})
  end

  return
end

-- TODO: Think of workflow to maximise usability for user
-- TODO: How to store settings? Ideally permanently?
-- TODO: Interesting code from example plugins:
--   - Get filename: https://github.com/xournalpp/xournalpp/blob/master/plugins/Export/main.lua#L29
--   - Toggle logic: https://github.com/xournalpp/xournalpp/blob/master/plugins/HighlightPosition/main.lua#L5
--   - UI: https://github.com/xournalpp/xournalpp/blob/master/plugins/MigrateFontSizes/main.lua
--   - OS interaction: https://github.com/xournalpp/xournalpp/blob/master/plugins/QuickScreenshot/main.lua
