"""Régénère versions.lock.json (empreintes des prompts, identifiants des modèles).

Usage :  python -m scripts.lock_versions
"""

import json

from maintidoc.versioning import write_lock

if __name__ == "__main__":
    print(json.dumps(write_lock(), indent=2, ensure_ascii=False))
