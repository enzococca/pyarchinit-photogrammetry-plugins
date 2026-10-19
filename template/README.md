# Plugin template

The start of a [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
plugin (1.1 or later). It adds two menu actions (a summary of the chunk, the statistics of the DEM
computed in the background), a command (`ppg plugin chunk-summary PROJECT`) and an export format
(*File > Esporta > Riepilogo del chunk (Markdown)*). The guide is
[docs/plugins.md](https://github.com/enzococca/pyarchinit-photogrammetry/blob/main/docs/plugins.md).

1. Copy this folder and rename the package (`ppg_my_plugin`), the distribution (`name` in
   `pyproject.toml`) and the entry point (`my-plugin`: the name of the plugin, lowercase letters,
   digits and dashes).
2. Install it in the Python environment of the program, editable, and run the tests:

   ```bash
   pip install -e .
   pytest -q tests
   ppg plugin                  # the command is listed
   ppg-gui                     # the menu Plugin has the actions
   ```

3. Replace what the plugin does. Keep the rules of the guide: only modules the program carries
   (`dependencies = []`; the packaged application has no pip), widgets only in the actions and in
   `done`, never in the function given to `context.run` or in an exporter, `progress(...)` called
   often in long loops.
4. Build the wheel (`python -m build --wheel`), attach it to a release of your repository and open
   a pull request adding it to the [index](../README.md#submitting-a-plugin).

Licence: GPL-2.0-or-later, like the program the plugin runs in.
