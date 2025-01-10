# Usage

The usage of the project is fairly simple. First, there is a Python script that performs the actual work & is useful for headless operations like batch processing. Second, and probably much more useful for the average user, the Lua plugin can be used from within Xournal++ and invokes the aforementioned Python script under the hood.

## The Lua plugin

Details relevant for usage of the Lua plugin:

1. Make sure to save your file in Xournal++ beforehand. The plugin will also let you know that you need to save your file first.
2. After installation, navigate to `Plugin > Xournal++ HTR` to invoke the plugin. Then select a filename and press `Save`. Lastly, wait a wee bit until the process is finished; the Xournal++ UI will block while the plugin applies HTR to your file. If you opened Xournal++ through a command-line, you can see progress bars that show the HTR process in real-time.

Note: Currently, the Xournal++ HTR plugin requires you to use a nightly build of Xournal++ because it uses upstream Lua API features that are not yet part of the stable build. Using the officially provided Nightly AppImag, see [here](https://xournalpp.github.io/installation/linux/), is very convenient. The plugin has been tested with the following nightly Linux build of Xournal++:

```
xournalpp 1.2.3+dev (583a4e47)
└──libgtk: 3.24.20
```

## The Python script

It is located in `xournalpp_htr/run_htr.py` and is features a command line interface that documents the usage of the Python script.