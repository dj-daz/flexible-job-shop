import math

import gym
import random


def conveyorMove(indexes, conveyor_size):
    for i in range(len(indexes)):
        temp = indexes[i] - 1
        if temp == -1:
            indexes[i] = conveyor_size-1
        else:
            indexes[i] = temp
    return indexes


class Agent:
    def __init__(self, factory, job_id, abilities, action_space, agent_statuses, agent_view, conveyor_size, job_sequence, op_sequence):
        self.ACTION_SPACE = action_space
        self.STATUSES = agent_statuses

        self.ID = job_id
        self.ABILITIES = abilities  # dictionary abilities and speed of abilities relative to each other .
        self.FACTORY = factory  # factory that agent is a part of
        self.WINDOW_SIZE = agent_view
        self.ADJUSTMENT = -1 * math.floor(self.WINDOW_SIZE / 2)
        self.CONVEYOR_SIZE = conveyor_size
        self.JOB_SEQUENCE = job_sequence.copy()
        self.OP_SEQUENCE = op_sequence.copy()
        self.tracker = 0
        self.seq = {"id": job_sequence[0], "op": op_sequence[0]}

        self.agentStatus = self.STATUSES["idle"]  # 0 = idle, 1 = starting job, 2 = processing job, 3 = finishing job
        self.current_ability = None  # ability in current use
        self.step_time_left = None  # time left of step
        self.step_total_time = None  # total time required for step
        self.job = None  # job assigned to agent to work on
        self.look_job = [None for _ in range(agent_view)]  # job agent is currently considering
        self.downtime = 0

    def sequenceUpdate(self):
        self.tracker += 1
        if self.tracker >= len(self.JOB_SEQUENCE):
            self.seq["id"] = -1
            self.seq["op"] = -1
        else:
            self.seq["id"] = self.JOB_SEQUENCE[self.tracker]
            self.seq["op"] = self.OP_SEQUENCE[self.tracker]

    def doAction(self, action, current_time):
        # print(f'agent {self.id} chose action {action} and is looking at: {self.look_job[0]}')
        if self.agentStatus == self.STATUSES["idle"]:
            self.downtime += 1
        if action == self.ACTION_SPACE[-1]:
            self.continueAction(current_time)  # continue processing job or continue idle if no job
        elif action == self.ACTION_SPACE[-2]:  # decline job
            self.declineJob()  # if job on agent, unload job
        else:
            self.acceptJob(current_time, action)

    def continueAction(self, current_time):
        if self.job is not None:  # has job, currently working on job
            # if job being processed, continue processing
            if self.job.getJobStatus() == self.job.STATUSES["processing"]:
                self.agentStep(current_time)
        else:  # no job, agent continues idle state
            pass

    def acceptJob(self, current_time, job_index):
        if self.job is not None:  # has job that is waiting to be processed
            # # if job being processed, continue processing
            # if self.job.getJobStatus() == self.job.STATUSES["processing"]:
            #     self.agentStep(current_time)
            if job_index != 0:  # if job not within agent reach
                self.declineJob()
            elif self.job.getJobStatus() == 0:  # if job is waiting, start job
                self.startJob(job_index)
        if self.job is None:  # has no job, sees a job
            pos = self.FACTORY.getAgentPos()
            pos = pos[self.ID]
            if job_index != 0:  # if job not within agent reach
                self.declineJob()
            else:
                self.FACTORY.addToConveyor(-1, index=pos)  # remove job from conveyor
                self.startJob(0)

    def startJob(self, job_index):
        self.job = self.look_job[job_index]
        # print(f'agent: {self.ID} looking at index {job_index}, job: {self.job}')
        self.job.jobStarted(self.ID)  # mark job as being processed

        self.step_time_left = self.job.current_step_time
        self.step_total_time = self.job.current_step_time
        self.agentStatus = self.STATUSES["starting job"]  # update agent status
        self.current_ability = self.job.current_op  # store current ability in use

    def declineJob(self):
        if self.job is None:  # if agent has no job do nothing
            self.agentReset()
            return

        pos = self.FACTORY.getAgentPos()
        pos = pos[self.ID]
        self.FACTORY.addToConveyor(1, find=True, index=pos,
                                   job=self.job)  # add job back to conveyor if not finished
        # reset variables
        self.agentReset()

    def finishStep(self, current_time):
        job_complete = self.job.stepComplete(current_time)  # set job step to complete
        self.sequenceUpdate()

        if job_complete:
            self.FACTORY.addToCompletedJobs(self.job)
            self.agentReset()
        else:  # reset, but keep job
            self.step_time_left = None
            self.step_total_time = None
            self.agentStatus = self.STATUSES["ready for unload"]
            self.current_ability = None

    def agentStep(self, current_time):
        self.step_time_left -= 1  # decrease time left
        self.agentStatus = self.STATUSES["processing job"]
        if self.step_time_left == 0:  # step complete
            self.finishStep(current_time)

    def getStatus(self):
        # return agent status
        return self.agentStatus

    def isWorking(self):
        return ((self.agentStatus == self.STATUSES["starting job"]) or
                (self.agentStatus == self.STATUSES["processing job"]))

    def agentReset(self):
        self.current_ability = None
        self.agentStatus = self.STATUSES["idle"]  # 0 = idle, 1 = starting job, 2 = processing job, 3 = finishing job
        self.step_time_left = None
        self.step_total_time = None
        self.job = None
        self.look_job = [None] * self.WINDOW_SIZE


class Job:
    def __init__(self, id, deadline, process, times, statuses):

        self.STATUSES = statuses

        self.ID = id  # job id
        self.DEADLINE = deadline  # job deadline
        self.PROCESS = process  # sequence of all steps
        self.TIMES = times  # estimated time for each step

        self.jobStatus = self.STATUSES["waiting"]  # job status - waiting for processing
        self.process_left = process.copy()  # sequence of remaining steps
        # self.times_left = times.copy()
        self.current_step = 0
        self.current_op = self.PROCESS[0]  # current step requried to be executed
        self.current_step_times = self.TIMES.query('step == 0').values.tolist()[0]
        self.agent_id = None
        self.current_step_time = None  # time for current step
        self.completed_time = None
        self.time_loaded = None
        self.op = 1

    def jobStarted(self, agent_id):
        self.agent_id = agent_id
        self.current_step_time = self.current_step_times[self.agent_id]
        self.jobStatus = self.STATUSES["processing"]

    def stepComplete(self, current_time):
        self.process_left.pop(0)  # remove completed step from list
        # self.times_left.pop(0)
        self.current_step += 1
        self.op += 1

        if not self.process_left:  # if step = to len of processes, all steps have been completed
            self.jobStatus = self.STATUSES["finished"]  # mark status as job finished
            self.completed_time = current_time
            return True

        self.current_op = self.process_left[0]  # update step required to be worked on
        self.current_step_times = self.TIMES.query(f'step == {self.current_step}').values.tolist()[0]
        self.current_step_time = None  # update time required for step

        self.jobStatus = self.STATUSES["waiting"]  # mark status as waiting to be worked on

        return False

    def getJobStatus(self):
        return self.jobStatus

    def __str__(self):
        return f'Job ID: {self.ID} \n Job Process: {self.PROCESS} \n Job Timings: {self.TIMES} \n ' \
               f'Job Deadline: {self.DEADLINE}'
        # return f'Job ID: {self.id}'

    def calculateDelay(self):
        return self.completed_time - self.DEADLINE

    def calculateLeadtime(self):
        return self.completed_time - self.time_loaded


def getMask(jobs, agent):
    # print(f'jobs {jobs}')
    mask = [True if el is not None else False for el in jobs]

    if not any(mask): # if all jobs are none
        mask.append(False)  # cannot take decline action
        mask.append(True)  # Take continue action
    elif agent.isWorking(): # if agent is working
        mask = [False for _ in jobs]
        mask.append(False)  # cannot take decline action
        mask.append(True)  # Take continue action
    else:  # if agent has to make a decision about a job
        i = 0
        for job in jobs:
            if job is not None:
                if (job.ID == agent.seq["id"]) and (job.op == agent.seq["op"]):
                    mask[i] = True
                else:
                    mask[i] = False
            i += 1
        mask.append(True)  # can take decline action
        mask.append(False)  # cannot take continue action

    if not any(mask):
        print(f'mask for agent {agent.ID} is {mask}')
    return mask


class Factory(gym.Env):
    def __init__(self, no_agents, no_abilities, abilities, conveyor_size, conveyor_fill, conveyor_pos, episode_steps,
                 no_jobs, action_space, job_statuses, agent_statuses, operation_size, window_size, agent_speeds,
                 job_sequence, op_sequence):
        super(Factory, self).__init__()

        self.ACTION_SPACE = action_space
        self.JOB_STATUSES = job_statuses
        self.AGENT_STATUSES = agent_statuses
        self.AGENT_SPEEDS = agent_speeds

        self.agents = []
        for i in range(no_agents):
            # initialise agents - abilities, time taken for each ability
            self.agents.append(Agent(self, i, abilities[i], action_space, agent_statuses, window_size, conveyor_size,
                                     job_sequence[i], op_sequence[i]))

        self.NUMBER_ABILITIES = no_abilities
        self.CONVEYOR_SIZE = conveyor_size  # size of conveyor belt
        self.CONVEYOR_FILL = conveyor_fill
        self.EPISODE_STEPS = episode_steps  # number of steps in each episode #####
        self.INIT_POS = conveyor_pos[:]  # initial position of agents and entry point
        self.NUMBER_JOBS = no_jobs
        self.OPERATION_SIZE = operation_size
        self.WINDOW_SIZE = window_size
        self.ADJUSTMENT = -1 * math.floor(self.WINDOW_SIZE / 2)

        self.conveyor = [None for _ in range(conveyor_size)]  # conveyor that holds jobs
        self.conveyor_buffer = []  # jobs that cannot fit on conveyor
        self.conveyor_pos = conveyor_pos[:]  # position of entry point and agents
        self.jobs_on_conveyor = 0  # number of jobs on the conveyor
        self.completed = []  # list of completed jobs
        self.job_completed = 0

        self.penalty = -1
        self.current_time = 0
        self.decline = 0

    def getAgents(self):
        return self.agents

    def getProcessObservation(self, job, agent):
        n = self.OPERATION_SIZE
        process_observation = job.process_left
        #process_observation = [2 if process in agent.ABILITIES else 1 for process in job.process_left]
        #print(f'agent {agent.ID} looking at job {job.ID} with process {job.process_left}, obs: {process_observation}')
        if len(process_observation) >= n:
            process_observation = process_observation[0:n]
        else:
            process_observation = process_observation + [0] * (n - len(process_observation))
        #print(f'agent {agent.ID} looking at job {job.ID} with process {job.process_left}, obs: {process_observation}')
        return process_observation

    def getObservations(self):
        # get machine status of all machines - shared observation
        global_observation = []
        machine_status = []
        observations = []
        masks = []

        for agent in self.agents:
            machine_status.append(agent.getStatus())

        global_observation = global_observation + machine_status

        # get job information - local observation
        for agent in self.agents:
            a_id = 10 * (agent.ID + 1)
            agent_id = [a_id]
            current_time = [self.current_time]
            jobs_left = [len(self.completed)]
            current_ability = [agent.current_ability if agent.current_ability else 0]
            agent_observation = []
            agent_global_observation = []
            agent_ob = []
            agent_observation = agent_id + machine_status + current_ability + agent_observation
            agent_global_observation = agent_global_observation + agent_id + machine_status
            if agent.getStatus() is self.AGENT_STATUSES["idle"]:  # observe conveyor belt if agent idle
                pos = self.conveyor_pos[agent.ID + 1]
                agent.look_job = []
                for i in range(self.WINDOW_SIZE):
                    job = self.conveyor[pos]
                    # print(f'agent: {agent.ID} looking at pos: {pos} at job {job}')
                    if job is None:  # if no job, observe no job
                        agent.look_job.append(None)
                        agent_observation = agent_observation + [0 for _ in range(self.OPERATION_SIZE)]
                        agent_ob = agent_ob + [0 for _ in range(self.OPERATION_SIZE)]
                    else:  # observe job process (n steps, if job seen)
                        agent.look_job.append(job)
                        process_observation = self.getProcessObservation(job, agent)
                        agent_observation = agent_observation + process_observation
                        agent_ob = agent_ob + process_observation
                    pos -= 1
                    if pos < 0:
                        pos = self.CONVEYOR_SIZE - 1
            #  else agent is busy working or deciding whether to unload so cannot make observation of jobs on conveyor
            else:
                agent.look_job = []
                agent.look_job.append(agent.job)
                if agent.getStatus() is self.AGENT_STATUSES["ready for unload"]: # allow agent to make observation when deciding to unload
                    process_observation = self.getProcessObservation(agent.job, agent)
                    agent_observation = agent_observation + process_observation
                    agent_ob = agent_ob + process_observation

                    pos = self.conveyor_pos[agent.ID + 1]
                    pos -= 1
                    if pos < 0:
                        pos = self.CONVEYOR_SIZE - 1

                    for i in range(self.WINDOW_SIZE - 1):
                        job = self.conveyor[pos]
                        # print(f'agent: {agent.ID} looking at pos: {pos} at job {job}')
                        if job is None:  # if no job, observe no job
                            agent.look_job.append(None)
                            agent_observation = agent_observation + [0 for _ in range(self.OPERATION_SIZE)]
                            agent_ob = agent_ob + [0 for _ in range(self.OPERATION_SIZE)]
                        else:  # observe job process (n steps, if job seen)
                            agent.look_job.append(job)
                            process_observation = self.getProcessObservation(job, agent)
                            agent_observation = agent_observation + process_observation
                            agent_ob = agent_ob + process_observation
                        pos -= 1
                        if pos < 0:
                            pos = self.CONVEYOR_SIZE - 1
                else:
                    process_observation = [0 for _ in range(self.OPERATION_SIZE)]
                    agent_observation = agent_observation + process_observation
                    agent_ob = agent_ob + process_observation
                    for i in range(self.WINDOW_SIZE - 1):
                        agent.look_job.append(None)
                        agent_observation = agent_observation + [0 for _ in range(self.OPERATION_SIZE)]
                        agent_ob = agent_ob + [0 for _ in range(self.OPERATION_SIZE)]
            global_observation = global_observation + agent_ob

            agent_mask = getMask(agent.look_job, agent)
            masks.append(agent_mask)
            observations.append(agent_observation)
            machine_status = machine_status[1:] + [machine_status[0]]
        #print(observations)
        return global_observation, observations, masks

    def step(self, actions, old_mask, old_states, current_time, show, job=None):

        self.current_time += 1
        for agent in self.agents:
            agent.doAction(actions[agent.ID], current_time)  # each agent to do action
        # job addition
        if job:
            self.conveyor_buffer.append(job)
        # if nothing on entry point and conveyor has space add to conveyor
        if (self.conveyor[self.conveyor_pos[0]] is None) and\
                (self.jobs_on_conveyor < (self.CONVEYOR_FILL * self.CONVEYOR_SIZE)) and self.conveyor_buffer:
            temp = self.conveyor_buffer.pop(0)
            self.conveyor[self.conveyor_pos[0]] = temp  # add job to conveyor from buffer
            self.addToConveyor(1)
            temp.time_loaded = current_time

        self.conveyor_pos = conveyorMove(self.conveyor_pos, self.CONVEYOR_SIZE)  # move conveyor one step
        # get reward for each agent
        rewards, dones = self._getReward(current_time, old_mask, old_states, actions, show)
        # get observations - shared machine states, local job observations
        if show:
            self.display()
        global_observation, observations, masks = self.getObservations()
        # print(observations, masks)
        return global_observation, observations, masks, rewards, dones

    def _getReward(self, current_time, mask, states, actions, show=False):
        # define reward values
        # WORKING_REWARD = 1/(2*self.EPISODE_STEPS)
        # WORKING_REWARD = 1/(self.EPISODE_STEPS)
        WORKING_REWARD = -1/self.EPISODE_STEPS
        dones = [False for _ in range(len(self.agents))]

        steps_complete = 0
        step_started = 0

        rewards = []

        share_reward = False
        for agent in self.agents:

            #self.penalty = self.penalty * pow(0.9, self.job_completed)

            #reward = self.penalty*10
            reward = -1.1

            #reward += self.job_completed*10
            if len(self.completed) >= self.NUMBER_JOBS:
                dones[agent.ID] = True
                # reward = 2*(1 - current_time/self.EPISODE_STEPS) * ((current_time - agent.downtime) / current_time)
                # reward = 2 * (1 - current_time / self.EPISODE_STEPS) + (agent.downtime / current_time)
                # reward = 2 * (1 - current_time / self.EPISODE_STEPS) * (1 - (len(agent.ABILITIES)/self.NUMBER_ABILITIES))
                # reward = (2) * (1 - (current_time / (self.EPISODE_STEPS)))
                # reward = 10 - current_time*(10/self.EPISODE_STEPS)
                # reward = -current_time
                # print(f'agent {agent.ID} received reward {reward}')
                # reward += 22
                # reward += 40
                # reward += 10
                # reward = -current_time
                # print(f'decline: {self.decline}')
                # reward += -current_time*2
            elif current_time >= self.EPISODE_STEPS - 1:
                # reward += -current_time*2
                pass

            if agent.isWorking():
                # reward = WORKING_REWARD * (1 - (len(agent.ABILITIES)/self.NUMBER_ABILITIES))
                # reward = -1*WORKING_REWARD
                # reward = 0
                op = agent.current_ability
                op_speed_list = self.AGENT_SPEEDS.loc[op].tolist()
                speed_sum = sum(op_speed_list)
                agent_speed = op_speed_list[agent.ID]
                reward += (agent_speed/speed_sum)  # bigger reward the faster the agent
            elif (actions[agent.ID] != self.ACTION_SPACE[-1]) and (actions[agent.ID] != self.ACTION_SPACE[-2]):
                act = actions[agent.ID]
                ind = self.OPERATION_SIZE * (1+act) + 1
                # print(f'states is {states[agent.ID]}, act is {actions[agent.ID]}')
                op = states[agent.ID][ind]
                op_speed_list = self.AGENT_SPEEDS.loc[op].tolist()
                speed_sum = sum(op_speed_list)
                agent_speed = op_speed_list[agent.ID]
                reward += ((agent_speed / speed_sum) / (2**act))  # bigger reward the faster the agent

            if agent.getStatus() == self.AGENT_STATUSES["starting job"]:
                step_started += 1
                reward += 0

            if agent.getStatus() == self.AGENT_STATUSES["ready for unload"]:
                steps_complete += 1
                # reward += 0.2
                #print(f'oh hi')
            if mask[agent.ID][-2] and (agent.getStatus() == self.AGENT_STATUSES["idle"]): # agent declines job
                reward += 0
                if show:
                    pass
                    #print(f'agent {agent.ID} declined job')
                #op = states[agent.ID][-self.OPERATION_SIZE*self.WINDOW_SIZE]
                #speed_list = self.AGENT_SPEEDS.loc[op].tolist()
                #print(f'agent_id {agent.ID} op: {op} speed_list {speed_list}')
                #if max(speed_list) != speed_list[agent.ID]:
                    #print(f'rewarded decline for agent {agent.ID}')
                #    reward += 0
                #    self.decline += 1
            if all(mask[agent.ID]) and (agent.getStatus() == self.AGENT_STATUSES["starting job"]):
                #op = states[agent.ID][-self.OPERATION_SIZE*self.WINDOW_SIZE]
                #speed_list = self.AGENT_SPEEDS.loc[op].tolist()
                #print(f'agent_id {agent.ID} op: {op} speed_list {speed_list}')
                #if max(speed_list) == speed_list[agent.ID]:
                    #print(f'rewarded accept for agent {agent.ID}')
                #    reward += 0
                pass
            if agent.getStatus() == self.AGENT_STATUSES["processing job"]:
                reward += 0
            # reward += self.job_completed * 10

            rewards.append(reward)

        for i in range(len(rewards)):
            rewards[i] = rewards[i] + steps_complete * 1 + step_started * 0
        #     # rewards[i] = rewards[i] + 0 * steps_complete
        #     if steps_complete > 0:
        #         pass  #print(f'{(self.EPISODE_STEPS - current_time)/self.EPISODE_STEPS}')
        self.job_completed = 0

        return rewards, dones

    def getAgentPos(self):
        return self.conveyor_pos[1:]

    def addToConveyor(self, add, find=False, index=None, job=None):
        self.jobs_on_conveyor += add

        if find:
            i = index
            while self.conveyor[i] is not None:
                i += 1
                if i == self.CONVEYOR_SIZE:
                    i = 0
            self.conveyor[i] = job
            return
        elif add == -1:
            self.conveyor[index] = None
            return

    def addToCompletedJobs(self, job):
        self.job_completed += 1
        self.completed.append(job)

    def reset(self):
        # reset agents
        for agent in self.agents:
            agent.agentReset()
            agent.downtime = 0

        self.conveyor = [None for _ in range(self.CONVEYOR_SIZE)]
        self.conveyor_buffer = []
        self.conveyor_pos = self.INIT_POS[:]
        self.jobs_on_conveyor = 0
        self.completed = []
        self.penalty = -1
        self.current_time = 0
        self.decline = 0

    def display(self):
        agent_places = ['.' for _ in range(self.CONVEYOR_SIZE)]
        agent_places[self.conveyor_pos[0]] = 'E'
        i = 0
        for pos in self.conveyor_pos[1:]:
            agent_places[pos] = f'A{self.agents[i].ID}' if self.agents[i].getStatus() == 0 else str(i) + str(self.agents[i].job.ID)
            i += 1
        display = [str(job.ID) if job else '.' for job in self.conveyor]

        print(*agent_places, sep='   ')
        print(*display, sep='   ')
        print('\n')

    def render(self, mode='human'):
        raise NotImplementedError
