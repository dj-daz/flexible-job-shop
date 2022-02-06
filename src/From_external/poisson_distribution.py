import random
import math
import pandas as pd


def arrival_time(rate, no_jobs, init_arrival_time=0):
    time = init_arrival_time
    times = [init_arrival_time]
    for i in range(1, no_jobs):
        inter_arrival_time = random.expovariate(rate)
        time += inter_arrival_time
        times.append(int(time))
    return times


def create_operation_v_machine(no_machines, max_operations, machine_abilities, min_max, large_no=-1):
    o_v_m = {}
    for j in range(0, no_machines):
        o_v_m[j] = [random.uniform((1.0 - min_max), (1.0 + min_max)) if i in machine_abilities[j] else large_no for i in
                    range(0, max_operations)]
    temp = pd.DataFrame(data=o_v_m)
    # print(temp.loc[2].tolist())
    return temp


def create_job_library(no_jobs, max_ops, max_steps, op_v_m, min_time, max_time):
    library_of_jobs = {}
    product_id = 0
    for i in range(no_jobs):
        # ops = random.uniform(2, max_ops)
        ops = random.randint(2, max_steps)
        process = random.choices(range(0, max_ops), k=ops)
        op = 0
        for step in process:
            base_time = random.uniform(min_time, max_time)
            time_ratios = op_v_m.loc[step].tolist()
            times = [None if tim == -1 else int(base_time/tim)*1 for tim in time_ratios]
            op_name = str(op) + '_' + str(step)
            library_of_jobs[(i, op, step)] = times
            op += 1
    job_lib = pd.DataFrame(library_of_jobs).T
    job_lib.index.names = ["prod_id", "step", "ops"]
    return job_lib


# print(arrival_time(5, 100))
# op_v_m = create_operation_v_machine(3, 4, [[0, 1, 2], [1, 2], [0, 3]], 0.5)
# print(op_v_m)
# job_lib = create_job_library(5, 4, 5, op_v_m, 2, 10)
# print(job_lib)
# job_1 = job_lib.query('prod_id == 1')
# job_1 = job_1.query('step == 1').values.tolist()[0]
# print(job_1)
# #print(job_1[:1])
# #print(job_1.index.get_level_values('ops').tolist())
