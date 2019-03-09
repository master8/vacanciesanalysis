#!/usr/bin/env bash

source activate master8_env
export PYTHONPATH="/home/mluser/master8_projects/pycharm_project_755/"
cd /home/mluser/master8_projects/pycharm_project_755/matching/
chmod +x $1
nohup python $1 $2 &

#Show processes
ps -A | grep python
#kill -9 8979798