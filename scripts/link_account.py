"""Émet un code de liaison à usage unique (10 min) pour associer un compte Telegram.

Usage (administrateur) :  python -m scripts.link_account tech.dupont
Le technicien envoie ensuite au bot :  /start <code>
"""

import argparse

from maintidoc.identity import IdentityStore

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("user_id")
    args = parser.parse_args()
    code = IdentityStore().issue_link_code(args.user_id)
    print(f"Code de liaison pour {args.user_id} (valable 10 min, usage unique) : {code}")
    print(f"Dans Telegram : /start {code}")
