import numpy as np
import time
from multiprocessing import Pool
import pyopencl as cl
import csv


# Generate a list of n random numbers
def generate_random_numbers(n, max_value):
    return np.random.randint(2, max_value, size=n, dtype=np.int64)


# Rabin-Miller primality test function (CPU-based)
def is_prime_cpu(n, k):
    n = int(n)  # Convert to Python's native integer type
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n < 2:
        return False

    s, d = 0, n - 1
    while d % 2 == 0:
        s += 1
        d //= 2

    for _ in range(k):
        a = np.random.default_rng().integers(2, n - 1, endpoint=True)
        a = int(a)  # Convert to Python's native integer type
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for r in range(s):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


# Sequential CPU processing
def sequential_prime_test(numbers, k):
    return [is_prime_cpu(n, k) for n in numbers]


# Multiprocessing CPU processing
def parallel_prime_test(numbers, k):
    with Pool(processes=None) as pool:
        return pool.starmap(is_prime_cpu, [(n, k) for n in numbers])


# GPU-based processing using OpenCL
def gpu_prime_test(numbers, k, kernel_code):
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices()
    context = cl.Context(devices)
    queue = cl.CommandQueue(context)
    program = cl.Program(context, kernel_code).build()

    numbers_np = np.array(numbers, dtype=np.int64)
    results_np = np.zeros_like(numbers_np, dtype=np.int8)
    random_numbers = np.random.randint(2, numbers_np.max(), size=(len(numbers_np) * k,), dtype=np.int64)

    numbers_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=numbers_np)
    results_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, results_np.nbytes)
    random_numbers_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=random_numbers)

    program.is_prime_gpu(queue, (len(numbers_np),), None, numbers_buf, results_buf, np.uint32(k), random_numbers_buf).wait()
    cl.enqueue_copy(queue, results_np, results_buf)
    queue.finish()

    return list(results_np)


# Timing function
def time_function(func, *args):
    start_time = time.time()
    results = func(*args)
    end_time = time.time()
    return results, end_time - start_time


# OpenCL kernel code
kernel_code = """
long pow_mod(long base, long exponent, long modulus) {
    long result = 1;
    base = base % modulus;

    while (exponent > 0) {
        if (exponent % 2 == 1)
            result = (result * base) % modulus;

        exponent = exponent >> 1;
        base = (base * base) % modulus;
    }

    return result;
}

__kernel void is_prime_gpu(__global const long* numbers, __global char* results, const unsigned int k, __global const long* random_numbers) {
    int idx = get_global_id(0);
    long n = numbers[idx];
    if (n == 2 || n == 3) {
        results[idx] = 1;
        return;
    }
    if (n % 2 == 0 || n < 2) {
        results[idx] = 0;
        return;
    }

    long s = 0;
    long d = n - 1;
    while (d % 2 == 0) {
        s += 1;
        d /= 2;
    }

    for (int i = 0; i < k; ++i) {
        long a = random_numbers[i]; 
        long x = pow_mod(a, d, n);
        if (x == 1 || x == n - 1)
            continue;

        int should_continue = 0;
        for (long r = 0; r < s; ++r) {
            x = pow_mod(x, 2, n);
            if (x == n - 1) {
                should_continue = 1;
                break;
            }
        }

        if (!should_continue) {
            results[idx] = 0;
            return;
        }
    }

    results[idx] = 1;
}
"""


def write_to_csv(file_name, data):
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)


if __name__ == '__main__':
    k = 5  # Number of tests in Rabin-Miller
    max_value = 10**12
    start_n = 100  # Starting value of n

    for i in range(4):  # 10 different values of n
        n = start_n * (10 ** i)  # Increase n by a factor of 10 each iteration
        print(f"\nTesting for n = {n}")

        for _ in range(3):
            numbers = generate_random_numbers(n, max_value)

            # # GPU processing
            # gpu_results, gpu_time = time_function(gpu_prime_test, numbers, k, kernel_code)
            # print(f"GPU processing time: {gpu_time} seconds")
            # write_to_csv('gpu_times.csv', [n, gpu_time])

            # Sequential CPU processing
            seq_results, seq_time = time_function(sequential_prime_test, numbers, k)
            print(f"Sequential CPU processing time: {seq_time} seconds")
            write_to_csv('sequential_times.csv', [n, seq_time])

            # # Multiprocessing CPU processing
            # mp_results, mp_time = time_function(parallel_prime_test, numbers, k)
            # print(f"Multiprocessing CPU processing time: {mp_time} seconds")
            # write_to_csv('multiprocessing_times.csv', [n, mp_time])

