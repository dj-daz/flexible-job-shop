from src.From_external.poisson_distribution import *

def get_job_lib(d, speed_seed=5, job_lib_seed=6):
    random.seed(speed_seed)
    agent_speeds = create_operation_v_machine(d['NUMBER_AGENTS'], d['MAX_OPERATIONS'], d['ABILITIES'], d['min_max'], d['MIN_OP_TIME'], d['MAX_OP_TIME'])
    print(agent_speeds)
    random.seed(job_lib_seed)
    job_library = create_job_library(d['unique_no_job'], d['MAX_OPERATIONS'], d['max_steps'], agent_speeds,
                                     d['MIN_OP_TIME'], d['MAX_OP_TIME'])
    return job_library, agent_speeds


def get_job_instance(d, instance_seed=None):
    random.seed(instance_seed)
    if instance_seed is None:
        random.seed()
    return random.choices(range(0, d['unique_no_job']), k=d['JOBS_TO_GENERATE'])


def get_arrival_times(d):
    random.seed()
    return arrival_time(d['rate'], d['JOBS_TO_COMPLETE'])

def job_arrived(job_instance, job_library, job_id):
    prod_id = job_instance[job_id]
    product = job_library.query(f'prod_id == {prod_id}')
    ops = product.index.get_level_values('ops').tolist()
    return ops, product


def to_GA_format(process, agent_no, abilities, prod):
    job = []

    j = 0
    for operation in process:
        op = []
        step_times = prod.query(f'step == {j}').values.tolist()[0]
        j+=1
        for i in range(agent_no):
            if operation in abilities[i]:
                op.append({'machine': i, 'processingTime': int(step_times[i])})
        job.append(op)
    return job