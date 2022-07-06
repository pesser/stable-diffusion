import time
from eden.client import Client
from eden.datatypes import Image

## set up a client
c = Client(url="http://0.0.0.0:5656", username="abraham")

## define input args to be sent
config = {
    "prompt": "a pink schoolbus underwater",
    "n_samples": 1,
    "ddim_steps": 200
}

# start the task
run_response = c.run(config)

print(run_response)

# check status of the task, returns the output too if the task is complete
results = c.fetch(token=run_response["token"])
print(results)

# one eternity later
#time.sleep(10)

## check status again, hopefully the task is complete by now
while True:
    print("FETCH")
    results = c.fetch(token=run_response["token"])
    print(results)
    # if 'output' in results:
    #     if results['output']:
    #         results['output']['creation1'].save('progress.png')
    #         idxr+=1
    time.sleep(5)


