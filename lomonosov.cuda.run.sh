#!/bin/bash

module add impi/5.0.1

sbatch -n $1 --ntasks-per-node=2 -p gputest -t 0-00:02:00 -o "./results/time.$1.$2.txt" impi ./cudirch $2 $2 $3

module rm impi/5.0.1
