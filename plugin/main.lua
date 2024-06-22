function initUi()
  app.registerUi({["menu"] = "Xournal++ HTR", ["callback"] = "run", ["accelerator"] = "<Control>F1"});
end
  
function run()
  os.execute("/home/martin/anaconda3/envs/xournalpp_htr/bin/python /home/martin/Development/xournalpp_htr/xournalpp_htr/xournalpp_htr.py -if /home/martin/Development/xournalpp_htr/tests/test_1.xoj -m dummy -of /home/martin/Development/xournalpp_htr/tests/test_1_from_Xpp.pdf")
  return
end

-- TODO: Think of workflow to maximise usability for user
-- TODO: How to store settings? Ideally permanently?
-- TODO: Interesting code from example plugins:
--   - Get filename: https://github.com/xournalpp/xournalpp/blob/master/plugins/Export/main.lua#L29
--   - Toggle logic: https://github.com/xournalpp/xournalpp/blob/master/plugins/HighlightPosition/main.lua#L5
--   - UI: https://github.com/xournalpp/xournalpp/blob/master/plugins/MigrateFontSizes/main.lua
--   - OS interaction: https://github.com/xournalpp/xournalpp/blob/master/plugins/QuickScreenshot/main.lua

-- function exampleCallback()
--   result = app.msgbox("Test123", {[1] = "Yes", [2] = "No"});
--   print("result = " .. result)
-- end