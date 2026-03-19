#!/bin/bash
FENICS_LIB=/home/darve/junlin25/miniconda3/envs/fenics/lib
export LD_LIBRARY_PATH=$FENICS_LIB
cd /home/darve/junlin25/MathDT-LLM/code/laser3d
exec /home/darve/junlin25/miniconda3/envs/fenics/bin/python3.11 laser_single_track.py
