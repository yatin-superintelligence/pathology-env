# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Pathology Env Environment.

This module creates an HTTP server that exposes the PathologyEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import PathologyAction, PathologyObservation
    from .pathology_env_environment import PathologyEnvironment
except (ImportError, ModuleNotFoundError):
    from models import PathologyAction, PathologyObservation
    from server.pathology_env_environment import PathologyEnvironment


# Create the app with web interface and README integration
app = create_app(
    PathologyEnvironment,
    PathologyAction,
    PathologyObservation,
    env_name="pathology_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

# Remove OpenEnv's default "/" redirect so our custom landing page takes priority
app.routes[:] = [r for r in app.routes if not (hasattr(r, 'path') and r.path == '/' and hasattr(r, 'methods') and 'GET' in r.methods)]

# Custom landing page at / — Gradio playground stays at /web/
@app.get("/", include_in_schema=False)
async def root():
    from fastapi.responses import HTMLResponse
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Blood Pathology LIMS Environment</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0d1117; color: #e6edf3; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
  .container { max-width: 720px; padding: 48px 36px; }
  h1 { font-size: 2rem; margin-bottom: 8px; }
  .badge { display: inline-block; background: #238636; color: #fff; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; margin-bottom: 24px; }
  .desc { color: #8b949e; font-size: 1.05rem; line-height: 1.6; margin-bottom: 32px; }
  h2 { font-size: 1.1rem; color: #58a6ff; margin-bottom: 12px; }
  .endpoints { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px 20px; margin-bottom: 28px; }
  .ep { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #21262d; font-size: 0.95rem; }
  .ep:last-child { border-bottom: none; }
  .ep .method { color: #7ee787; font-family: monospace; font-weight: 600; }
  .ep .path { color: #e6edf3; font-family: monospace; }
  .ep .info { color: #8b949e; }
  .tasks { display: flex; gap: 12px; margin-bottom: 28px; }
  .task { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 20px; flex: 1; text-align: center; text-decoration: none; color: inherit; transition: border-color 0.2s; }
  .task:hover { border-color: #58a6ff; }
  .task .level { font-weight: 700; font-size: 1rem; }
  .task.easy .level { color: #7ee787; }
  .task.medium .level { color: #d29922; }
  .task.hard .level { color: #f85149; }
  .task .variants { color: #8b949e; font-size: 0.8rem; margin-top: 4px; }
  .scores { background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; margin-bottom: 28px; }
  .scores table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  .scores th { background: #21262d; padding: 8px 12px; text-align: left; color: #8b949e; font-weight: 600; }
  .scores td { padding: 8px 12px; border-top: 1px solid #21262d; }
  .scores tr:hover { background: #1c2128; }
  .playground-btn { display: block; text-align: center; background: #238636; color: #fff; padding: 14px 28px; border-radius: 8px; font-size: 1rem; font-weight: 600; text-decoration: none; margin-bottom: 28px; transition: background 0.2s; }
  .playground-btn:hover { background: #2ea043; }
  .footer { color: #484f58; font-size: 0.8rem; text-align: center; margin-top: 16px; }
  a { color: #58a6ff; text-decoration: none; }
</style>
</head>
<body>
<div class="container">
  <h1>🩸 Blood Pathology LIMS Environment</h1>
  <span class="badge">● Running</span>
  <p class="desc">High-fidelity clinical pathology diagnostic environment. The agent interprets blood biomarker panels against context-adjusted reference ranges, cross-references medications for drug-lab interactions, and issues ICD-10–coded diagnostic reports.</p>

  <a href="/web/" class="playground-btn">🧪 Open Interactive Playground</a>

  <h2>Task Difficulty</h2>
  <div class="tasks">
    <div class="task easy"><div class="level">Easy</div><div class="variants">3 variants</div></div>
    <div class="task medium"><div class="level">Medium</div><div class="variants">3 variants</div></div>
    <div class="task hard"><div class="level">Hard</div><div class="variants">2 variants</div></div>
  </div>

  <h2>Baseline Scores</h2>
  <div class="scores">
    <table>
      <tr><th>Model</th><th>Easy</th><th>Medium</th><th>Hard</th><th>Avg</th></tr>
      <tr><td>Gemma-4-26B-A4B</td><td>0.99</td><td>0.94</td><td>0.08</td><td><b>0.67</b></td></tr>
      <tr><td>Gemma-4-31B</td><td>0.99</td><td>0.94</td><td>0.30</td><td><b>0.74</b></td></tr>
      <tr><td>Qwen 3.6 Plus</td><td>0.99</td><td>0.51</td><td>0.99</td><td><b>0.83</b></td></tr>
      <tr><td>MiniMax M2.7</td><td>0.99</td><td>0.94</td><td>0.33</td><td><b>0.75</b></td></tr>
    </table>
  </div>

  <h2>API Endpoints</h2>
  <div class="endpoints">
    <div class="ep"><span><span class="method">GET</span> <span class="path">/health</span></span><span class="info">Health check</span></div>
    <div class="ep"><span><span class="method">POST</span> <span class="path">/step</span></span><span class="info">Execute action</span></div>
    <div class="ep"><span><span class="method">POST</span> <span class="path">/reset</span></span><span class="info">Start new episode</span></div>
    <div class="ep"><span><span class="method">GET</span> <span class="path">/state</span></span><span class="info">Current state</span></div>
    <div class="ep"><span><span class="method">GET</span> <span class="path">/schema</span></span><span class="info">Action/observation schema</span></div>
  </div>

  <p style="text-align:center; margin-top:20px;"><a href="https://huggingface.co/spaces/yatin-superintelligence/pathology-env/blob/main/README.md" target="_blank" style="color:#58a6ff; font-size:0.95rem;">📄 Full Documentation (README)</a></p>

  <p class="footer">Built by <a href="https://www.yatintaneja.in">Yatin Taneja</a> · <a href="https://www.imsuperintelligence.ai">IM Superintelligence</a></p>
</div>
</body>
</html>"""
    return HTMLResponse(content=html)




def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m pathology_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn pathology_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
