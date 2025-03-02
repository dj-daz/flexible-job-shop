#!/usr/bin/env python

# This script contains a high level overview of the proposed hybrid algorithm
# The code is strictly mirroring the section 4.1 of the attached paper

import os
import csv
import time
from datetime import datetime



import torch
import yaml

from src.From_external.DynamicFactoryEnvironment import *
from src.utils import gantt
from src.genetic import encoding, decoding, genetic, termination
from src import config
from src.From_external.helper import *

# ================================================================================================================ #
with open("src/From_external/parameters.yaml", "r") as yamlfile:
    d = yaml.load(yamlfile, Loader=yaml.FullLoader)

# File naming
suffix = d['suffix']
folder = d['folder']
trained_folder = d['trained_folder']
log_to_file = d['log_to_file']

# episode dyn_parameters
EPISODES = d['EPISODES']
EPISODE_STEPS = d['EPISODE_STEPS']
SHOW_EVERY = d['SHOW_EVERY']
SHOW = d['SHOW']
TEST_EPISODE = d['TEST_EPISODE']

# DQN dyn_parameters
double = d['double']
PER = d['PER']

BATCH_SIZE = d['BATCH_SIZE']
BUFFER_SIZE = d['BUFFER_SIZE']
TARGET_UPDATE_FREQ = d['TARGET_UPDATE_FREQ']
TAU = d['TAU']
LEARNING_FREQ = d['LEARNING_FREQ']
epsilon = d['epsilon']  # 0 means no exploration, 1 means only exploration
EPSILON_DECAY = d['EPSILON_DECAY']
MIN_EPSILON = d['MIN_EPSILON']
LEARNING_RATE = d['LEARNING_RATE']
DISCOUNT = d['DISCOUNT']
LEARNING_START = d['LEARNING_START']

# job dyn_parameters
DYNAMIC = d['DYNAMIC']
MAX_DEADLINE = d['MAX_DEADLINE']
MAX_OPERATIONS = d['MAX_OPERATIONS']
MIN_OP_TIME = d['MIN_OP_TIME']
MAX_OP_TIME = d['MAX_OP_TIME']

# environment dyn_parameters
n = d['n']   # number of upcoming processes seen
m = d['m']  # window size

FRAME_SKIP = d['FRAME_SKIP']

JOBS_TO_COMPLETE = d['JOBS_TO_COMPLETE']
JOBS_TO_GENERATE = d['JOBS_TO_GENERATE']
CONVEYOR_SIZE = d['CONVEYOR_SIZE']
CONVEYOR_POS = d['CONVEYOR_POS']
CONVEYOR_FILL = d['CONVEYOR_FILL']

NUMBER_AGENTS = d['NUMBER_AGENTS']
ABILITIES = d['ABILITIES']

OB_SIZE = 1 + NUMBER_AGENTS + 1 + n*m  # agent id + each agent status + current ability being used + upcoming operations of each job in window
ACT_SIZE = m + 1 + 1 # accept action for each window and a decline action and a continue action

ACTION_SPACE = [i for i in range(ACT_SIZE)]
JOB_STATUSES = d['JOB_STATUSES']
AGENT_STATUSES = d['AGENT_STATUSES']

average_score = d['average_score']
best_score = d['best_score']

rate = d['rate']
min_max = d['min_max']
unique_no_job = d['unique_no_job']
max_steps = d['max_steps']

SPEED_SEED = d['SPEED_SEED']
JOB_LIB_SEED = d['JOB_LIB_SEED']
INSTANCE_SEED = d['INSTANCE_SEED']

# initialise environment
job_library, agent_speeds = get_job_lib(d, speed_seed=SPEED_SEED, job_lib_seed=JOB_LIB_SEED)
print(f'job_library is: {job_library}')
job_instance = get_job_instance(d, instance_seed=INSTANCE_SEED)
print(f'job instance is: {job_instance}')

random.seed()

max_job = int(CONVEYOR_SIZE * CONVEYOR_FILL)

iterations = math.ceil(JOBS_TO_COMPLETE / max_job)

# ==================================================================================================================== #
# metric logs
if log_to_file:
    if not os.path.isdir(f'{folder}/'):
        os.makedirs(folder)
    file_name_metric = "{}/Metrics_Ep{}_A{}_J{}_n{}_m{}_{}.csv".format(folder, EPISODES, NUMBER_AGENTS, JOBS_TO_COMPLETE, n, m, suffix)
    metric_file = open(file_name_metric, "w")
    metric_writer = csv.writer(metric_file)
    metric_writer.writerow(["Episode", "Steps", "Average Delay", "Average Lead Time", "M_1 Downtime", "M_2 Downtime",
                           "M_3 Downtime"])

    file_name_rewards = "{}/Rewards_Ep{}_A{}_J{}_n{}_m{}_{}.csv".format(folder, EPISODES, NUMBER_AGENTS, JOBS_TO_COMPLETE, n, m, suffix)
    rewards_file = open(file_name_rewards, "w")
    rewards_writer = csv.writer(rewards_file)
    header = ["Agent" + str(i) for i in range(NUMBER_AGENTS)]
    header = header + ["Total"]
    rewards_writer.writerow(header)

# episode metrics
episode_rewards = {}  # dictionary that stores rewards of each player for each episode
total_episode_rewards, episode_number, episode_steps, episode_delays, episode_lead_times = [], [], [], [], []
episode_downtimes, loss_list = [[], [], []], [[], [], []]
# ==================================================================================================================== #

jobsFact = []
jobsGA = []

for i in range(JOBS_TO_COMPLETE):
    ops, product = job_arrived(job_instance, job_library, i)
    job_incoming = Job(i, MAX_DEADLINE, ops, product, JOB_STATUSES)
    jobGA = to_GA_format(ops, NUMBER_AGENTS, ABILITIES, product)

    jobsFact.append(job_incoming)
    jobsGA.append(jobGA)

print(f'jobsGA: {jobsGA}')
last_in = 0
job_sequences = [[] for i in range(NUMBER_AGENTS)]
operation_sequences = [[] for i in range(NUMBER_AGENTS)]

# Parameters Setting
for i in range(iterations):
    job_section = jobsGA[:]
    parameters = {'machinesNb': NUMBER_AGENTS, 'jobs': jobsGA[last_in:last_in+max_job]}

    t0 = time.time()

    # Initialize the Population
    population = encoding.initializePopulation(parameters)
    gen = 1

    # Evaluate the population
    while not termination.shouldTerminate(population, gen):
        # Genetic Operators
        population = genetic.selection(population, parameters)
        population = genetic.crossover(population, parameters)
        population = genetic.mutation(population, parameters)

        gen = gen + 1

    sortedPop = sorted(population, key=lambda cpl: genetic.timeTaken(cpl, parameters))

    t1 = time.time()
    total_time = t1 - t0
    print("Finished in {0:.2f}s".format(total_time))

    # Termination Criteria Satisfied ?
    machine_operations, job_sequence, operation_sequence = decoding.decode(parameters, sortedPop[0][0], sortedPop[0][1])

    gantt_data, job_sequence, operation_sequence = decoding.translate_decoded_to_gantt(machine_operations)

    job_sequence = [[el + last_in for el in a] for a in job_sequence]

    job_sequences = [job_sequences[i] + job_sequence[i] for i in range(NUMBER_AGENTS)]
    operation_sequences = [operation_sequences[i] + operation_sequence[i] for i in range(NUMBER_AGENTS)]

    # print(sortedPop[0][0], sortedPop[0][1])

    if config.latex_export:
        gantt.export_latex(gantt_data)
    else:
        gantt.draw_chart(gantt_data)

    last_in = last_in+max_job

print(f'job sequence is: {job_sequences}')
print(f'operation sequence is: {operation_sequences}')
print(f'machine operations is: {machine_operations}')
# job_sequences = [[2, 1, 5, 5, 0, 8, 8, 7], [1, 0, 0, 5, 4, 3, 8], [0, 0, 2, 2, 1, 1, 9, 6]]
# operation_sequences = [[1, 2, 2, 3, 5, 1, 3, 1], [1, 3, 4, 1, 1, 1, 2], [1, 2, 2, 3, 3, 4, 1, 1]]
# ==================================================================================================================== #

factory = Factory(NUMBER_AGENTS, d['MAX_OPERATIONS'], ABILITIES, CONVEYOR_SIZE, CONVEYOR_FILL, CONVEYOR_POS, EPISODE_STEPS, JOBS_TO_COMPLETE,
                  ACTION_SPACE, JOB_STATUSES, AGENT_STATUSES, n, m, agent_speeds, job_sequences, operation_sequences)


# track total training time
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
print("============================================================================================")

for episode in range(EPISODES):

    if episode % SHOW_EVERY == 0:
        print(f'Episode: {episode + 1}')

    factory.reset()
    _, states, masks = factory.getObservations()

    job_id = 0

    for episode_step in range(EPISODE_STEPS):
        actions = []
        job_incoming = None
        if job_id < JOBS_TO_COMPLETE:
            ops, product = job_arrived(job_instance, job_library, job_id)
            job_incoming = Job(job_id, MAX_DEADLINE, ops, product, JOB_STATUSES)
            job_id += 1

        for agent in factory.getAgents():
            state = states[agent.ID]
            if not torch.is_tensor(state):
                state = torch.FloatTensor(state)
            mask = masks[agent.ID]
            for i in range(len(mask)):
                if mask[i]:
                    action = i
                    break
            actions.append(action)

        show = (episode % SHOW_EVERY == 0) and SHOW
        _, new_states, new_masks, rewards, dones = factory.step(actions, masks, states, episode_step, show, job_incoming)

        states = new_states
        masks = new_masks

        if any(dones):
            episode_number.append(episode)
            episode_steps.append(episode_step)
            delay_sum = 0
            lead_sum = 0
            # calculate delay and lead times
            #for job in jobs:
            #    if job.jobStatus == JOB_STATUSES["finished"]:
            #        delay_sum = delay_sum + job.calculateDelay()
            #        lead_sum = lead_sum + job.calculateLeadtime()

            if episode % SHOW_EVERY == 0:
                print(f'ending episode on step: {episode_step}')

            if log_to_file:
                try:
                    metric_writer.writerow([episode_number[-1], episode_steps[-1], episode_delays[-1], episode_lead_times[-1],
                                            episode_downtimes[0][-1], episode_downtimes[1][-1], episode_downtimes[2][-1]])
                except:
                    print(episode_number)
                    print(episode_steps)
                    print(episode_delays)
                    print(episode_lead_times)
                    print(episode_downtimes)

            break



    if episode % SHOW_EVERY == 0:
        print(f'completed jobs: {len(factory.completed)}')
        print("============================================================================================")

print("============================================================================================")
end_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
print("Finished training at (GMT) : ", end_time)
print("Total training time  : ", end_time - start_time)
print("============================================================================================")
