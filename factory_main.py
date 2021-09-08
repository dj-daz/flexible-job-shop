#!/usr/bin/env python

# This script contains a high level overview of the proposed hybrid algorithm
# The code is strictly mirroring the section 4.1 of the attached paper
import math
import random
import sys
import time

from src.utils import parser, gantt
from src.genetic import encoding, decoding, genetic, termination
from src import config

# ================================================================================================================ #

# episode parameters
EPISODES = 1
EPISODE_STEPS = 300
SHOW_EVERY = 1
SHOW = False


# environment parameters
CONVEYOR_SIZE = 12
NUMBER_AGENTS = 3
NUMBER_ABILITIES = 4
ABILITIES = [[1, 2, 3], [2, 3], [1, 4]]
CONVEYOR_POS = [0, 1, 5, 9]
OB_SIZE = 6
ACT_SIZE  = 2
JOBS_TO_COMPLETE = 20

# ================================================================================================================ #

def generateJob(id, current_time, max_time, number_abillities, max_steps, min_step_time, max_step_time, abilities, agent_no):
    LEEWAY = 16
    LEEWAY2 = 20
    k = random.randint(1, max_steps)
    process = random.choices(range(1, number_abillities + 1), k=k)
    times = [random.randint(min_step_time, max_step_time) for i in range(len(process))]
    job_time = sum(times)

    deadline = random.randint(current_time + job_time + LEEWAY, current_time + job_time + LEEWAY+LEEWAY2)
    if deadline > max_time:
        deadline = max_time

    job = []

    j = 0
    for operation in process:
        op = []
        for i in range(agent_no):
            if operation in abilities[i]:
                op.append({'machine': i, 'processingTime': times[j]})
        j += 1
        job.append(op)

    return job, deadline

def generateJobV2(id, current_time, max_time, number_abillities, max_steps, min_step_time, max_step_time, abilities, agent_no):
    LEEWAY = 16
    LEEWAY2 = 20
    k = random.randint(1, max_steps)
    process = random.choices(range(1, number_abillities + 1), k=k)
    max_times = []
    job = []
    for operation in process:
        op = []
        times = []
        for i in range(agent_no):
            if operation in abilities[i]:
                times.append(random.randint(min_step_time, max_step_time))
                op.append({'machine': i, 'processingTime': times[-1]})
        job.append(op)
        max_times = max(times)

    job_time = sum(max_times)

    deadline = random.randint(current_time + job_time + LEEWAY, current_time + job_time + LEEWAY+LEEWAY2)
    if deadline > max_time:
        deadline = max_time

    return job, deadline

jobs = []
random.seed(100)
for i in range(math.floor(JOBS_TO_COMPLETE)):
    job, deadline = generateJob(i, i * 2, 100, 4, 5, 2, 5, ABILITIES, NUMBER_AGENTS)
    jobs.append(job)
# Parameters Setting
parameters = {'machinesNb': NUMBER_AGENTS, 'jobs': jobs}

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
gantt_data = decoding.translate_decoded_to_gantt(decoding.decode(parameters, sortedPop[0][0], sortedPop[0][1]))

if config.latex_export:
    gantt.export_latex(gantt_data)
else:
    gantt.draw_chart(gantt_data)
