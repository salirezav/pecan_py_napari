# Using pecan_py (pecan-video) as a separate library

This project uses **[pecan-video](https://pypi.org/project/pecan-video/)** (PyPI package name; import name `pecan_py`) for video/image processing. The library is published at https://pypi.org/project/pecan-video/

**Install:** `pip install pecan-video`

You can develop the library in its own repo and use a local path override during development (see below).

---

## 1. Separate repo (done)

The library lives in its own repo and is published to PyPI as **pecan-video** so that:

- You version and release it independently.
- Others can install with `pip install pecan-video`.
- Import in Python: `from pecan_py import PecanVideo` (module name stays `pecan_py`).

---

## 2. Package pecan_py for pip install

### 2.1 Add packaging to the pecan_py folder

In your pecan_py repo, add a `pyproject.toml` so it is an installable package (published as **pecan-video** on PyPI).

Copy the template from this project: [pecan_py_pyproject.toml.example](pecan_py_pyproject.toml.example), or create `pyproject.toml` in the **root** of the repo with content like:

```toml
[project]
name = "pecan-video"
version = "0.1.0"
description = "Pecan video and image processing utilities"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "BSD-3-Clause" }
dependencies = [
    "numpy",
    "opencv-python",
    "Pillow",
    "scipy",
    "tqdm",
]

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["pecan_py*"]
```

Layout in that repo:

- Either **flat**: `pecan_py/` (your current module) and `pyproject.toml` at repo root.
- Or **src layout**: `src/pecan_py/` and `where = ["src"]` in `pyproject.toml`.

Imports stay `import pecan_py` / `from pecan_py.PecanVideo import PecanVideo`; the installable package name on PyPI is **pecan-video**.

### 2.2 Install locally in editable mode

From the pecan_py repo root:

```bash
uv sync
# or: pip install -e .
```

Then others (or this napari project) can install it from a path or from Git (see below).

---

## 3. Publish to PyPI (done)

The package is published as **pecan-video** at https://pypi.org/project/pecan-video/

For new releases: bump `version` in the library’s `pyproject.toml`, then from the repo root run `uv build` and `uv publish` (or use [Trusted Publishers](https://docs.pypi.org/trusted-publishers/) with GitHub Actions).

---

## 4. Use pecan-video in this napari plugin project

### Option A: From PyPI (default)

This project depends on **pecan-video**. Install everything with:

```bash
uv sync --all-extras
```

Use in code:

```python
from pecan_py import PecanVideo
# or
from pecan_py.PecanVideo import PecanVideo
```

### Option B: Local development (editable override)

To work on the pecan-video library and this plugin at the same time, add a **source override** in this repo so uv uses your local clone (editable):

1. **Folder layout** (e.g. both repos on Desktop):

   ```
   Desktop/
   ├── pecan_py/              # pecan-video library repo (with pyproject.toml)
   └── pecan_py_napari/       # this repo
   ```

2. **In this repo’s `pyproject.toml`**, add:

   ```toml
   [tool.uv.sources]
   pecan-video = { path = "../pecan_py", editable = true }
   ```

   Then run `uv sync --all-extras`. Changes in the library are picked up after you restart napari. Remove this section when you’re not doing local library development so installs use PyPI again.

3. Run the plugin:

   ```bash
   uv sync --all-extras
   uv run napari
   ```

### Option C: Install from GitHub

Users can install directly from the library repo if you host it on GitHub:

```bash
pip install git+https://github.com/your-username/pecan_py.git
```

For a specific tag: `pip install git+https://github.com/your-username/pecan_py.git@v0.1.0`

---

## 5. Summary

| Goal | Action |
|------|--------|
| Install library | `pip install pecan-video` · [PyPI](https://pypi.org/project/pecan-video/) |
| Use in this plugin | Already a dependency; run `uv sync --all-extras`. |
| Develop both at once | Add `[tool.uv.sources]` with path to your pecan_py repo (see Option B above). |

The template `pyproject.toml` for the library is in this repo: **docs/pecan_py_pyproject.toml.example** (use package name `pecan-video` when publishing).
