#!/bin/bash

git add .
git commit -m "Quick update: $(date '+%Y-%m-%d %H:%M:%S')"
git push origin $(git branch --show-current)
echo "✅ 代码已推送"
