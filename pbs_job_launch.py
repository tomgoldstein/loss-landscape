"""A script for launching jobs using the PB queue manager"""

__author__ = 'Tom'


import os
import random
import string
import subprocess
import time


QSUB_TEMPLATE = """
#####################################
#   PBS script auto generated
#   by pbs_job_launch.py
#####################################
#!/bin/sh -f
#$ -cwd
#PBS -N {name}
#PBS -o {logfile}
#PBS -e {logfile}.err
#PBS -l walltime={hours}:{mins}:{secs}
#PBS -A {account}
#PBS -l select={num_nodes}:mpiprocs={procs_per_node}:ncpus={cpus_per_node}{accelerator}
#PBS -q {queue}
#####################################
echo "-----------------command run by pbs_job_launch.py-----------------------"
echo {message}
echo "---------------------------pbs options----------------------------------"
echo PBS -l select={num_nodes}:mpiprocs={procs_per_node}:ncpus={cpus_per_node}
echo PBS -l walltime={hours}:{mins}:{secs}
echo PBS -A {account}
echo PBS -q {queue}
echo "------------------------------------------------------------------------"
echo "Job started on" `date`
echo "------------------------------------------------------------------------"

echo "{setup}"

{setup}

echo "{command}"

{command}

echo "------------------------------------------------------------------------"
echo "Job ended on" `date`
echo "------------------------------------------------------------------------"
"""

def submit_job(command, job_name="job",
                       outfile_prefix="",
                       outfile_name=None,
                       hours=0,
                       mins=0,
                       secs=0,
                       account="USNAM37766432",
                       num_nodes=1,
                       procs_per_node=1,
                       cpus_per_node=1,
                       accelerator='',
                       queue="debug",
                       delete_after_submit=False,
                       call_qsub=True,
                       setup_command=None,
                       verbose = False
                       ):

    rand_id = ''.join(random.sample(string.ascii_letters + string.digits, 9))
    #rand_id = str(int(round(time.time() * 1000))) # tag file with the time is was made
    if not outfile_name:
        outfile_name = job_name
    logfile = outfile_prefix+outfile_name+"_output_"+rand_id+".txt"
    script_name = outfile_prefix+outfile_name+"_qsub_" +rand_id + '.sh'

    if len(accelerator) > 0 and accelerator[0] != ':':

        accelerator = ':'+accelerator

    pbs_script = QSUB_TEMPLATE.format(
                                  message="\""+command.replace('\"',r'\"')+"\"",
                                  command=command,
                                  name=job_name,
                                  logfile=logfile,
                                  hours=hours,
                                  mins=mins,
                                  secs=secs,
                                  accelerator=accelerator,
                                  account=account,
                                  num_nodes=num_nodes,
                                  procs_per_node=procs_per_node,
                                  cpus_per_node=cpus_per_node,
                                  queue=queue,
                                  setup=setup_command)

    pbs_script+="\nexit\n"

    if verbose:
        print(pbs_script)

    if call_qsub:
        print("writing pbs script to file: %s" % script_name)
        with  open(script_name, 'w') as f:
            f.write(pbs_script)
        try:
            print("submitting (%s): "%queue+command)
            subprocess.call('qsub ' + script_name, shell=True)
        except:
            print("Error:  call to qsub failed")
    else:
        print("NOT submitting job (%s): " % queue + command)

    if delete_after_submit:
        print("deleting script: %s"%script_name)
        try:
            os.remove(script_name)
        except:
            print("ERROR: Unable to remove pbs script")
    print("")
