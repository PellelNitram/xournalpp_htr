## Usage

Details relevant for usage of the plugin:

1. Make sure to save your file beforehand. The plugin will also let you know that you
   need to save your file first.
2. After installation, navigate to `Plugin > Xournal++ HTR` to invoke the plugin. Then
   select a filename and press `Save`. Lastly, wait a wee bit until the process is
   finished; the Xournal++ UI will block while the plugin applies HTR to your file.
   If you opened Xournal++ through a command-line, you can see progress bars that show
   the HTR process in real-time.

Note: Currently, the Xournal++ HTR plugin requires you to use a nightly build of
Xournal++ because it uses upstream Lua API features that are not yet part of the
stable build. Using the officially provided Nightly AppImag, see
[here](https://xournalpp.github.io/installation/linux/), is very convenient.
The plugin has been tested with the following nightly Linux build of Xournal++:

```
xournalpp 1.2.3+dev (583a4e47)
└──libgtk: 3.24.20
```

Details relevant for development of the plugin:

1. Activate environment: ``conda activate xournalpp_htr``. Alternatively use ``source activate_env.sh`` as shortcut.
2. Use the code.
3. To update the requirements file: ``pip freeze > requirements.txt``.