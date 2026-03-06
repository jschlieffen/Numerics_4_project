import concurrent.futures
import time
from itertools import repeat

def func(params, x1, x2):
    
    print(f"params: {params}, x1: {x1}, x2: {x2}")
    time.sleep(1)
    return x1 + x2

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
        val = 0
        for res in pool.map(func, repeat(0, 6), range(0, 6), range(0, 12, 2)):
            val += res
        print(val)
        print(sum(range(0, 12, 2)) + sum(range(0, 6)))