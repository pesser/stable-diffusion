import os
import random
import string

from eden.block import Block
from eden.hosting import host_block

eden_block = Block()


my_args = {
    "prompt": "Hello world",
    
}

@eden_block.run(args=my_args)
def run_stable_diffusion(config):
    
    prompt = config["prompt"]
    
    # setup file paths
    
    return {
        "completion": 'hello world'
    }



host_block(
    block=eden_block,
    port=5656,
    redis_port=6379,
    redis_host="eden-dev-gene-redis",
    host="0.0.0.0",
    logfile="log.txt",
    log_level="debug",
    max_num_workers=4,
    requires_gpu=True,
)

#python3 server.py -n "1" -p "5656" -rh eden-dev-gene-redis -rp "6379"
