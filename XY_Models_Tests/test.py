import multitasking
import time
import random
import signal

multitasking.set_max_threads(multitasking.config["CPU_CORES"] * 2)
# kill all tasks on ctrl-c
signal.signal(signal.SIGINT, multitasking.killall)

# or, wait for task to finish on ctrl-c:
# signal.signal(signal.SIGINT, multitasking.wait_for_tasks)

@multitasking.task # <== this is all it takes :-)
def hello(count):
    sleep = random.randint(1,3)/2
    print("Hello %s (sleeping for %ss)" % (count, sleep))
    time.sleep(sleep)
    print("Goodbye %s (after for %ss)" % (count, sleep))


for i in range(0, 10):
    hello(i+1)
