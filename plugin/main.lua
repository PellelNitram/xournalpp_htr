function initUi()
  app.registerUi({["menu"] = "Xournal++ HTR", ["callback"] = "run", ["accelerator"] = "<Control>F1"});
end

function save_file(path)
  if path:len() > 0 then

    -- Read settings: I use this (https://stackoverflow.com/a/41176958). An
    -- alternative could have been https://stackoverflow.com/a/41176826. Both
    -- found using G"lua read settings file".
    local config = require "config"

    config.filename = '"' .. app.getDocumentStructure()['xoppFilename'] .. '"'
    config.output_file = '"' .. path .. '"'

    command = config.python_executable .. " " .. config.xournalpp_htr_path
              .. " -if " .. config.filename
              .. "  -m " .. config.model
              .. " -of " .. config.output_file
    if config.debug_HTR_command then
      print(command)
    else
      os.execute(command)
    end

  end
end

function run()

  document_structure = app.getDocumentStructure()

  if document_structure['xoppFilename']:len() == 0 then
    app.openDialog('Please save document prior to exporting it as searchable PDF!', {"Ok"}, "", true)
  else
    app.fileDialogSave("save_file", "untitled.pdf")
  end

end

-- TODO: Think of workflow to maximise usability for user
-- TODO: How to store settings? Ideally permanently?
-- TODO: Interesting code from example plugins:
--   - Get filename: https://github.com/xournalpp/xournalpp/blob/master/plugins/Export/main.lua#L29
--   - Toggle logic: https://github.com/xournalpp/xournalpp/blob/master/plugins/HighlightPosition/main.lua#L5
--   - UI: https://github.com/xournalpp/xournalpp/blob/master/plugins/MigrateFontSizes/main.lua
--   - OS interaction: https://github.com/xournalpp/xournalpp/blob/master/plugins/QuickScreenshot/main.lua
