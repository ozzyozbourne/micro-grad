# tinygrad internals: an interactive study guide

22 chapters on how tinygrad (commit 6a404871, v0.14.0) works internally. The pages include interactive demos, data captured
from real runs, and runnable examples.

- Serve it: `python3 server.py 8765` (static files plus `POST /api/trace`, which the live explorer in chapter 22 calls), then open `index.html`.
- Examples: `DEV=CPU python3 examples/<NN>_*.py` (needs `clang` and tinygrad at the pinned commit: `git clone https://github.com/tinygrad/tinygrad && cd tinygrad && git checkout 6a404871bbe527136c04997bbd3ea4abd323e2f3 && pip install -e .`). Exceptions:
  `17_python_emulator.py` needs `DEV=PYTHON NOOPT=1`, `17_hcq_null.py` needs `DEV=NULL`, `17_debug2.py` needs `DEBUG=2`, and `07_index_functions.py` needs `NOOPT=1`.
- `tools/trace.py` captures every pipeline stage of a snippet as JSON; `tools/uop2json.py` converts UOp graphs for the renderer.
- `data/` holds captured traces; most `examples/*` scripts that print data with `--dump`/`--json` regenerate it.
