import time
from eden.client import Client
from eden.datatypes import Image

## set up a client
c = Client(url="http://0.0.0.0:5656", username="abraham")

## define input args to be sent
config = {
    "prompt": "hello stable diffusion"
}

# start the task
run_response = c.run(config)

print(run_response)

# check status of the task, returns the output too if the task is complete
results = c.fetch(token=run_response["token"])
print(results)

# one eternity later
time.sleep(2)

## check status again, hopefully the task is complete by now
results = c.fetch(token=run_response["token"])
print(results)