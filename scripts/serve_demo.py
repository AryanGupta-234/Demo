from __future__ import annotations
import uvicorn
from pre_cab.api import create_app
from pre_cab.config import Settings
from pre_cab.provider_factory import build_provider


def main() -> None:
    settings = Settings.from_env()
    model = None
    if __import__('os').getenv('PRE_CAB_ENABLE_LLM', '0').lower() in {'1','true','yes'}:
        model = build_provider()
    app = create_app(model=model)
    uvicorn.run(app, host='127.0.0.1', port=8000)

if __name__ == '__main__':
    main()
