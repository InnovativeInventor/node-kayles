import subprocess
import time
import sys
from statistics import stdev

results = []
for i in range(int(sys.argv[1])):
    time.sleep(1)
    start = time.time()
    subprocess.run("./target/release/non-attacking-queens --size 8", shell=True)
    end = time.time()
    results.append(end-start)
    print(f"Run {i}/{int(sys.argv[1])} took {end-start}.")

print(f"Average runtime (n=len(results)) {sum(results)/len(results)} (stdev={stdev(results)}).")
