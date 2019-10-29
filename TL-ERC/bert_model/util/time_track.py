import time
from functools import partial


def base_time_desc_decorator(method, desc='test_description'):
    def timed(*args, **kwargs):

        # Print Description
        # print('#' * 50)
        print(desc)
        # print('#' * 50 + '\n')

        # Calculation Runtime
        start = time.time()

        # Run Method
        try:
            result = method(*args, **kwargs)
        except TypeError:
            result = method(**kwargs)

        # Print Runtime
        print('Done! It took {:.2} secs\n'.format(time.time() - start))

        if result is not None:
            return result

    return timed


def time_desc_decorator(desc): return partial(base_time_desc_decorator, desc=desc)


@time_desc_decorator('this is description')
def time_test(arg, kwarg='this is kwarg'):
    time.sleep(3)
    print('Inside of time_test')
    print('printing arg: ', arg)
    print('printing kwarg: ',  kwarg)


@time_desc_decorator('this is second description')
def no_arg_method():
    print('this method has no argument')


if __name__ == '__main__':
    time_test('hello', kwarg=3)
    time_test(3)
    no_arg_method()
