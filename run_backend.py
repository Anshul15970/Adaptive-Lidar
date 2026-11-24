import argparse
import sys
from pathlib import Path
import uvicorn

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Adaptive LiDAR Lab Backend Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable live code reloading")
    args = parser.parse_args()

    print(f"Starting Adaptive LiDAR Lab Backend on http://{args.host}:{args.port}")
    uvicorn.run("app.backend.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

