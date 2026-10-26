#!/bin/sh
# Download the Conjura multi-region eCommerce MMM dataset (CC BY 4.0) into data/raw/conjura.
# Anderson, A. (2024). figshare. https://doi.org/10.6084/m9.figshare.25314841.v3
set -eu
cd "$(dirname "$0")/.."
mkdir -p data/raw/conjura
curl -sSf "https://api.figshare.com/v2/articles/25314841" \
  | python3 -c 'import json,sys; [print(f["name"], f["download_url"], f["computed_md5"]) for f in json.load(sys.stdin)["files"]]' \
  | while read -r name url md5; do
      curl -sSfL -o "data/raw/conjura/$name" "$url"
      got=$(python3 -c 'import hashlib,sys; print(hashlib.md5(open(sys.argv[1],"rb").read()).hexdigest())' "data/raw/conjura/$name")
      [ "$got" = "$md5" ] || { echo "checksum mismatch for $name" >&2; exit 1; }
      echo "ok  $name"
    done
