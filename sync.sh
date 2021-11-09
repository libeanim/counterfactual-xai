#!/bin/bash

SOURCE=$(dirname "${BASH_SOURCE[0]}")
DST_BASE=ahochlehnert48@134.2.168.52:/home/bethge/ahochlehnert48/code
DST=$DST_BASE/counterfactual_xai

echo "SOURCE:      $SOURCE"
echo "DESTINATION: $DST"

# rsync -a --progress --itemize-changes --update --dry-run $SOURCE $DST
rsync -rh --progress --itemize-changes --exclude=sync.sh --exclude-from=.gitignore --exclude=.mypy_cache --delete $SOURCE $DST
