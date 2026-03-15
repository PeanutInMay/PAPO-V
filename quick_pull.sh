#!/bin/bash

git pull origin $(git branch --show-current)
echo "✅ 代码已更新"