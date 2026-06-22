"""Launch HTTPS tunnel + update .trex manifest with the public URL.

Usage:
    python tunnel.py              # auto-detect best method
    python tunnel.py --ngrok      # force ngrok (pyngrok)
    python tunnel.py --cloudflared # force cloudflared

Prerequisites:
    - pip install pyngrok  (recommended — zero install, auto-downloads ngrok)
    - OR cloudflared installed: winget install Cloudflare.cloudflared
    - Backend running: uvicorn main:app --reload --port 8000

What this does:
    1. Starts an HTTPS tunnel on port 8000
    2. Updates extension/text-to-viz.trex with the public HTTPS URL
    3. Prints instructions for adding the extension to Tableau Cloud
    4. On exit (Ctrl+C), restores .trex to localhost
"""

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

TREX_PATH = Path(__file__).parent / "extension" / "text-to-viz.trex"
LOCAL_PORT = 8000


def update_trex(url: str) -> None:
    """Replace the <url> in .trex manifest with the tunnel URL."""
    content = TREX_PATH.read_text(encoding="utf-8")
    new_content = re.sub(
        r"<url>.*?</url>",
        f"<url>{url}/extension/index.html</url>",
        content,
    )
    TREX_PATH.write_text(new_content, encoding="utf-8")
    print(f"[OK] Updated {TREX_PATH.name} -> {url}/extension/index.html")


def restore_trex() -> None:
    """Restore .trex to localhost on exit."""
    update_trex("http://localhost:8000")
    print("[OK] Restored .trex to http://localhost:8000")


def _print_instructions(tunnel_url: str) -> None:
    print()
    print("=" * 60)
    print(f"  TUNNEL ACTIVE: {tunnel_url}")
    print("=" * 60)
    print()
    print("Next steps in Tableau Cloud:")
    print(f"  1. Settings -> Extensions -> Enable Specific Extensions")
    print(f"  2. Add URL: {tunnel_url}")
    print(f"  3. Create/open a Dashboard -> drag 'Extension' object")
    print(f"  4. Choose 'Access Local Extensions'")
    print(f"  5. Upload: extension/text-to-viz.trex")
    print()
    print("Press Ctrl+C to stop the tunnel.")
    print()


# ── ngrok via pyngrok (recommended — pip install pyngrok) ────────────────


def run_ngrok():
    """Start tunnel via pyngrok — auto-downloads ngrok binary, no install needed."""
    try:
        from pyngrok import ngrok, conf
    except ImportError:
        print("pyngrok not installed. Run: python -m pip install pyngrok")
        sys.exit(1)

    print("Starting ngrok tunnel on port", LOCAL_PORT, "...")
    print("(Make sure uvicorn is running: uvicorn main:app --reload --port 8000)\n")

    tunnel = ngrok.connect(LOCAL_PORT, "http")
    tunnel_url = tunnel.public_url
    # Force https
    if tunnel_url.startswith("http://"):
        tunnel_url = tunnel_url.replace("http://", "https://", 1)

    update_trex(tunnel_url)
    _print_instructions(tunnel_url)

    try:
        # Keep alive until Ctrl+C
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down tunnel...")
    finally:
        ngrok.disconnect(tunnel.public_url)
        ngrok.kill()
        restore_trex()


# ── cloudflared ──────────────────────────────────────────────────────────


def _find_cloudflared() -> str:
    """Locate cloudflared executable."""
    found = shutil.which("cloudflared")
    if found:
        return found
    for candidate in [
        r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
        r"C:\Program Files\cloudflared\cloudflared.exe",
    ]:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "cloudflared not found. Install: winget install Cloudflare.cloudflared"
    )


def run_cloudflared():
    """Start tunnel via cloudflared quick tunnel."""
    cloudflared_bin = _find_cloudflared()
    print(f"Starting cloudflared tunnel on port {LOCAL_PORT} ...")
    print(f"Using: {cloudflared_bin}")
    print("(Make sure uvicorn is running: uvicorn main:app --reload --port 8000)\n")

    proc = subprocess.Popen(
        [cloudflared_bin, "tunnel", "--url", f"http://localhost:{LOCAL_PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        for line in proc.stdout:
            match = re.search(r"(https://[a-zA-Z0-9\-]+\.trycloudflare\.com)", line)
            if match:
                tunnel_url = match.group(1)
                update_trex(tunnel_url)
                _print_instructions(tunnel_url)
            sys.stdout.write(line)
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n\nShutting down tunnel...")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        restore_trex()


# ── main ─────────────────────────────────────────────────────────────────


def main():
    force = None
    if "--ngrok" in sys.argv:
        force = "ngrok"
    elif "--cloudflared" in sys.argv:
        force = "cloudflared"

    if force == "ngrok":
        run_ngrok()
    elif force == "cloudflared":
        run_cloudflared()
    else:
        # Auto-detect: prefer pyngrok (simpler), fallback to cloudflared
        try:
            import pyngrok  # noqa: F401
            print("[auto] Using ngrok (pyngrok)\n")
            run_ngrok()
        except ImportError:
            try:
                _find_cloudflared()
                print("[auto] Using cloudflared\n")
                run_cloudflared()
            except FileNotFoundError:
                print("No tunnel tool found. Install one:")
                print("  python -m pip install pyngrok    (recommended)")
                print("  winget install Cloudflare.cloudflared")
                sys.exit(1)


if __name__ == "__main__":
    main()
