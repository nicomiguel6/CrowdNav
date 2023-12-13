import numpy as np
import pysocialforce
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY
import toml
from pathlib import Path



class pySocialForce(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'PySocialForce'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.groups = None
        self.radius = 0.3
        self.max_speed = 5

        base_directory = Path(__file__).resolve().parents[3]
        self.config_file_path = base_directory / 'crowd_nav/configs/scene_config.toml'

        # Read the config file
        with open(self.config_file_path, 'r') as file:
            self.config_data = toml.load(file)

        self.sim = None

    def configure(self, config):
        # self.time_step = config.getfloat('orca', 'time_step')
        # self.neighbor_dist = config.getfloat('orca', 'neighbor_dist')
        # self.max_neighbors = config.getint('orca', 'max_neighbors')
        # self.time_horizon = config.getfloat('orca', 'time_horizon')
        # self.time_horizon_obst = config.getfloat('orca', 'time_horizon_obst')
        # self.radius = config.getfloat('orca', 'radius')
        # self.max_speed = config.getfloat('orca', 'max_speed')
        self.groups = config.get('pysocialforce', 'groups')
        return

    def set_phase(self, phase):
        return

    def predict(self, state):
        """
        Create a pysocialforce simulation at each time step and run one step
        PySocialForce API: 


        :param state:
        :return:
        """
        #state

        self_state = state.self_state
        
        # create full stacked state
        # initial_state = []
        # for human in state.human_states:
        #     temp = np.hstack(human)


        # numAgents = len(self.sim.get_states)
        # if self.sim is not None and numAgents != len(state.human_states):
        #     del self.sim
        #     self.sim = None
        if self.sim is None:
            self.sim = pysocialforce.Simulator(state, groups=self.groups, obstacles=None, config_file=self.config_file_path)
        else:
            self.sim.peds.state = np.hstack(self_state.position, self_state.velocity, self_state.get_goal_position()).tolist()
            
           
            # for i, human_state in enumerate(state.human_states):
            #     self.sim.setAgentPosition(i + 1, human_state.position)
            #     self.sim.setAgentVelocity(i + 1, human_state.velocity)

        self.sim.step()
        action = ActionXY(self.sim.peds.vel())
        self.last_state = state

        return action
