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


def create_operation_v_machine(no_machines, max_operations, machine_abilities, min_max, min_time, max_time, large_no=-1):
    o_v_m = {}
    avg = (min_time+max_time)/2.0
    lower = avg/max_time
    upper = avg/min_time
    for j in range(0, no_machines):
        o_v_m[j+1] = [avg/float(random.randint(min_time, max_time)) if i in machine_abilities[j] else large_no for i in range(1, max_operations+1)]
    temp = pd.DataFrame(data=o_v_m)
    temp.index = temp.index + 1
    # print(temp.loc[2].tolist())
    return temp


def create_job_library(no_jobs, max_ops, max_steps, op_v_m, min_time, max_time):
    library_of_jobs = {}
    product_id = 0
    for i in range(no_jobs):
        # ops = random.uniform(2, max_ops)
        no_ops = random.randint(2, max_steps)
        ops = random.choices(range(1, max_ops+1), k=no_ops)
        step = 0
        for op in ops:
            upper = max_time*0.1
            # base_time = random.uniform(min_time, max_time)
            base_time = (min_time + max_time) / 2
            time_ratios = op_v_m.loc[op].tolist()
            times = [None if tim == -1 else max(min_time, min(round(base_time/tim)*1, max_time)) for tim in time_ratios]
            library_of_jobs[(i, step, op)] = times
            step += 1
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
