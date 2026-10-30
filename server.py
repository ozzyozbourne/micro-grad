#!/usr/bin/env python3
"""Serves the study guide. Run: python3 server.py [port]  (default 8765).

GET  /...         static files (unchanged)
POST /api/trace   body {"code": str, "opts": {"noopt": bool, "beam": int}} -> trace JSON (see tools/trace.py).
                  Runs the snippet in a subprocess (DEV=CPU) with a 60 s timeout.
"""
import http.server, socketserver, sys, os, json, threading, importlib.util
ROOT = os.path.dirname(os.path.abspath(__file__))
TRACE_TIMEOUT = 60
MAX_CODE = 20000
_trace_slots = threading.BoundedSemaphore(2)   # at most 2 traces at once
_trace_mod = None

def _tracer():
  global _trace_mod
  if _trace_mod is None:   # load tools/trace.py by path (the name "trace" would shadow the stdlib module)
    spec = importlib.util.spec_from_file_location("tg_trace", os.path.join(ROOT, "tools", "trace.py"))
    _trace_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_trace_mod)
  return _trace_mod

class Handler(http.server.SimpleHTTPRequestHandler):
  def __init__(self, *a, **kw): super().__init__(*a, directory=ROOT, **kw)
  def end_headers(self):
    self.send_header("Cache-Control", "no-cache")
    super().end_headers()

  def _json(self, code:int, obj):
    body = json.dumps(obj, separators=(",", ":")).encode()
    self.send_response(code)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def do_POST(self):
    if self.path.split("?")[0] != "/api/trace": return self._json(404, {"ok": False, "error": "unknown endpoint"})
    try:
      n = int(self.headers.get("Content-Length", "0"))
      if n > 200000: return self._json(413, {"ok": False, "error": "request too large"})
      req = json.loads(self.rfile.read(n) or b"{}")
      code, opts = req.get("code", ""), req.get("opts", {}) or {}
      if not isinstance(code, str) or not code.strip(): return self._json(400, {"ok": False, "error": "missing 'code'"})
      if len(code) > MAX_CODE: return self._json(400, {"ok": False, "error": f"code longer than {MAX_CODE} chars"})
      if not isinstance(opts, dict): return self._json(400, {"ok": False, "error": "'opts' must be an object"})
    except Exception as e:
      return self._json(400, {"ok": False, "error": f"bad request: {e}"})
    if not _trace_slots.acquire(timeout=TRACE_TIMEOUT): return self._json(503, {"ok": False, "error": "tracer busy, try again"})
    try: res = _tracer().run_trace(code, opts, timeout=TRACE_TIMEOUT)
    except Exception as e: res = {"ok": False, "error": f"tracer failed: {type(e).__name__}: {e}"}
    finally: _trace_slots.release()
    self._json(200, res)

class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
  allow_reuse_address = True
  daemon_threads = True

if __name__ == "__main__":
  port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
  print(f"serving {ROOT} on :{port}")
  Server(("0.0.0.0", port), Handler).serve_forever()
