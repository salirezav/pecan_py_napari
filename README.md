# napari-pecan-py

A [napari](https://napari.org/) plugin for Pecan Py (npe2-based).

## Installation

Install with [uv](https://docs.astral.sh/uv/) (recommended) or pip:

```bash
# With uv (includes napari and Qt for running the GUI)
uv sync --all-extras

# Or with pip
pip install -e ".[all]"
```

## Development

The project is installed in **editable** mode, so code changes are picked up when you restart napari.

1. **Install dependencies and the plugin**
   ```bash
   uv sync --all-extras --group dev
   ```

2. **Run napari with the plugin**
   ```bash
   uv run napari
   ```

3. In napari, open **Plugins → Pecan Py** and click **Run** to try the sample widget.

4. After editing plugin code in `src/napari_pecan_py/`, restart napari (`uv run napari`) to see changes. No need to run `uv sync` again unless you change dependencies.

## Project structure

```
src/napari_pecan_py/
├── __init__.py
├── _version.py
├── napari.yaml    # npe2 manifest (commands, widgets)
└── _widget.py     # widget implementation
```

## Tests

```bash
uv run pytest
```

## License

BSD-3-Clause
