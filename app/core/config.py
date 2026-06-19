import json
from pathlib import Path
from typing import Any


def load_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parent.parent / "config.json"
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Erreur : Le fichier {config_path} est introuvable.")
        return {}
    except json.JSONDecodeError:
        print(f"Erreur : Le fichier {config_path} est mal formaté.")
        return {}
