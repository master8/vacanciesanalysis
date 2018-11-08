#!/usr/bin/env bash

source activate master8_env
export PYTHONPATH="/home/mluser/master8_projects/pycharm_project_755/"
cd /home/mluser/master8_projects/pycharm_project_755/classification/
chmod +x main.py
nohup python $1 &

#Show processes
ps -A | grep python
#kill -9 8979798