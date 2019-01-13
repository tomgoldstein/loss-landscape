"""For launching jobs on the cluster"""

from pbs_job_launch import submit_job
import os
import time
import userdata


opts = {   "job_name":"plt",
            "hours":100, #72
           "mins":0,
           "secs":0,
           "account":userdata.get_account(),
           "num_nodes":1,
           "procs_per_node":1,
           "cpus_per_node":20,
           "queue":"standard",
           "delete_after_submit":False,  #  Delete the pbs shell script immediately after calling qsub?
           "call_qsub":True
        }


opts["setup_command"] = \
"""
source ~/.bashrc
cd /gpfs/scratch/tomg/loss-landscape
"""
opts["outfile_prefix"] = "/gpfs/scratch/tomg/loss-landscape/logs/"

nprocs = 16

command = "python plot_surface.py --ngpu 4  --cuda --model resnet56_noshort --x=-1.2:1.2:1001 --y=-1.2:1.2:1001 "\
 + "--model_file cifar10/trained_nets/resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7 --dir_type weights "\
 + "--xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot  --batch_size 2048 --threads 16 "\
 + "--data_split 5 --rank {rank} --of {nprocs} &> logs/plot_noshort56_{rank}_of_{nprocs}.log "

for rank in range(nprocs):
    specs={'rank':rank,
           'nprocs':nprocs}
    opts["job_name"] = "plt{rank}".format(**specs)
    submit_job(command.format(**specs), **opts)
