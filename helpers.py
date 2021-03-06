import time

def timer():
    '''
    A timer to be used as a decorator, printing time used for a function to run
    in microseconds.
    '''
    def timer(fnc):
        def inner(*arg):
            # inner function
            start = time.time()
            func_result = fnc(*arg)
            end = time.time()
            elapsed = (end - start) * 1000000
            print(f'Time elapsed: {str(elapsed)} microsec')
            return func_result

        return inner

    return timer

