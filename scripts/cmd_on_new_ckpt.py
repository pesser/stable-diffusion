import os
import subprocess
import time
import fire


class Checker(object):
    def __init__(self, filename, interval=60):
        self._cached_stamp = 0
        self.filename = filename
        self.interval = interval
    
    def check(self, cmd):
        while True:
            stamp = os.stat(self.filename).st_mtime
            if stamp != self._cached_stamp:
                self._cached_stamp = stamp
                print(f"{self.__class__.__name__}: Detected a new file at {self.filename}, running evaluation commands on it.")
                subprocess.run(cmd, shell=True)
            else:
                time.sleep(self.interval)


def run(filename, cmd):
    checker = Checker(filename, interval=60)
    checker.check(cmd)


if __name__ == "__main__":
    fire.Fire(run)
                

