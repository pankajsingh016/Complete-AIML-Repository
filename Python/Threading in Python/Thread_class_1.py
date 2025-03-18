import concurrent.futures
import threading
import time

start = time.perf_counter()

def do_something(seconds):
    print(f'Print we are sleeping for {seconds} second....')
    time.sleep(seconds)
    return 'Done Sleeping.....'

with concurrent.futures.ThreadPoolExecutor() as executor:
    # f1 = executor.submit(do_something,1)
    # f2 = executor.submit(do_something,1)
    # print(f1.result())
    # print(f2.result())
    secs = [5,4,3,2,1]
    result = [executor.submit(do_something,sec) for sec in secs]
    for f in concurrent.futures.as_completed(result):
        print(f.result())

    results = executor.map(do_something,secs)
    for result in results:
        print(result)
    




# t1 = threading.Thread(target=do_something)
# t2 = threading.Thread(target=do_something)

"""t1.start()
t2.start()

t1.join()
t2.join() # Make Code run in Sync
"""
#
# threads = []
#
#
# for _ in range(10):
#     t = threading.Thread(target=do_something,args=[1.5])
#     t.start()
#     threads.append(t)
#
# for thread in threads:
#     thread.join()

finish = time.perf_counter()
print(f'Finished in {round(finish-start,2)} second(s)')

"""
CPU Bound Process vs I/O Bound Processes
CPU bound tasks: Tasks which are utilizing cpu power
I/O bound tasks: Tasks which are waiting for process like input and output
Threading is beneficial for IO bound processes. It dosen't run the code at the same time but instead give the illusion of 
the it.
CPU Bound we use multiprocessing. 
Real-World Application
1. Downloading the data from mhultiple urls can be optimized
2. Any I/O Bound operations are run parallely by creating thread pool.
"""
