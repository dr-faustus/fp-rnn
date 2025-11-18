import uuid
import itertools
import os
from argparse import ArgumentParser
import subprocess
# import numpy as np

def make_submisison_file_content(executable, arguments, output, error, log, cpus=1, gpus=0, memory=1000, disk="1G", gpu_spec=None, min_mem=None):
    d = {
        'executable': executable,
        'arguments': arguments,
        'output': output,
        'error': error,
        'log': log,
        'request_cpus': cpus,
        'request_gpus': gpus,
        'request_memory': memory,
        'request_disk': disk
    }
    return d

def run_job(uid, bid, d):
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    job_file = os.path.join('tmp', uid)
    with open(job_file, 'w') as f:  
        for key, value in d.items():  
            f.write(f'{key} = {value}\n')
        f.write("queue")

    subprocess.run(["condor_submit_bid", str(bid), job_file]) 


if __name__ == '__main__':
    # sweep over quantities in the [] brakets
    # gpu_spec = 'NVIDIA A100-SXM4-40GB'
    # gpu_spec = 'NVIDIA A100-SXM4-40GB'

    uid = uuid.uuid4().hex[:10]
    arguments = f""
    # arguments = f"{group} {run[0]} {model} {run[1]} {run[2]} {run[3]} {run[4]} {curr_epochs_num} {run[6]} {run[7]} xent"
    output = f"runs/{uid}.stdout"
    error = f"runs/{uid}.stderr"
    log = f"runs/{uid}.log"
    cpus = 32
    gpus = 0 # requesting 1 GPU!!
    memory = 20 * 1024 # 40GB of memory
    disk = '20G' # 20GB of disk
    executable = "get_data.sh"

    try:
        content = make_submisison_file_content(executable, arguments, output, error, log, cpus, gpus=gpus, memory=memory, disk=disk)
        run_job(uid, 25, content) # SECOND ARGUMENT IS THE BID!
    except:
        raise ValueError("Crashed.")
    print("Done.")