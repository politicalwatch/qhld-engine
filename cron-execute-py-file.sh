#!/bin/bash
# USAGE: bash cron-execute-py-file projectpath filename fileinfo

echo "****************"
echo "Start $3 at $(date)"

cd $1
source .venv/bin/activate
python $2

echo "Finish $3 at $(date)"
echo "****************"
