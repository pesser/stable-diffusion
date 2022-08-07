import time
from eden.client import Client
from eden.datatypes import Image

## set up a client
c = Client(url="http://0.0.0.0:5656", username="abraham")


def test_generate():

    config = {
        "mode": "generate",
        "text_input": ["a pink schoolbus underwater"],
        "n_samples": 1,
        "ddim_steps": 200
    }

    # start the task
    run_response = c.run(config)
    print(run_response)

    # check status of the task, returns the output too if the task is complete
    results = c.fetch(token=run_response["token"])
    print(results)

    while True:
        results = c.fetch(token=run_response["token"])
        print(results)
        # if results["status"]["status"] == "complete":
        #     if results['output']:
        #         results['output']['creation'].save('results/final.png')
        time.sleep(5)



def test_interpolate():

    config = {
        "mode": "interpolate",
        "text_input": [
            "a pink schoolbus in outer space", 
            "a blue minivan in outer space", 
            "a spacheship in outer space", 
            "an alien in outer space"
        ],
        "n_interpolate": 5,
        "ddim_steps": 50
    }

    # start the task
    run_response = c.run(config)
    print(run_response)

    # check status of the task, returns the output too if the task is complete
    results = c.fetch(token=run_response["token"])
    print(results)

    while True:
        results = c.fetch(token=run_response["token"])
        print(results)
        time.sleep(5)



#test_generate()
test_interpolate()