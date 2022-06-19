
import time
import retro
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import math
import argparse
import os

from algos.agents.dqn_agent import DQNAgent
from algos.models.dqn_cnn import DQNCnn
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame
import pandas as pd

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--train', type=str2bool, default=True, help='Set to "True" for training, set to "False" for deploy the model')
parser.add_argument('--load_epoch', type=str, default='1000', help='Load which trained model')
opt = parser.parse_args()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

GAME_NAME = 'Kungfu-Nes'
TRIAL_NUMBER = '(50)'
env = retro.make(game=GAME_NAME, scenario=None)
possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())
env.seed(0)

print("The size of frame is: ", env.observation_space.shape)
print("No. of Actions: ", env.action_space.n)
env.reset()

INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = len(possible_actions)
SEED = 0
GAMMA = 0.99           # discount factor
BUFFER_SIZE = 1000000  # replay buffer size
BATCH_SIZE = 32        # Update batch size
LR = 0.00025           # learning rate
TAU = 1e-3             # for soft update of target parameters
UPDATE_EVERY = 50      # how often to update the network
REPLAY_AFTER = 50000   # After which threshold replay to be started
EPS_START = 1          # starting value of epsilon
EPS_END = 0.1          # Ending value of epsilon
EPS_DECAY = 250        # Rate by which epsilon to be decayed
TRAIN = opt.train      # option for training or testing
LOAD_EPOCH = opt.load_epoch
MOMENTUM = 0.95        # momentum of the RMS optimizer
FINAL_EXPLORATION = 100

MODEL_DIR = ['./models/{}-dqn-{}/policy_net'.format(GAME_NAME, TRIAL_NUMBER),
             './models/{}-dqn-{}/target_net'.format(GAME_NAME, TRIAL_NUMBER)] # directory to save and load policy model and target model respectively

# Check whether the specified path exists or not
if not os.path.exists('./models/{}-dqn-{}'.format(GAME_NAME, TRIAL_NUMBER)) :
    os.makedirs('./models/{}-dqn-{}'.format(GAME_NAME, TRIAL_NUMBER))

agent = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY,
                 REPLAY_AFTER, DQNCnn, TRAIN, LOAD_EPOCH, MODEL_DIR, MOMENTUM)

start_epoch = 0
scores = []
scores_window = deque(maxlen=20)
reward_window = deque(maxlen=20)

epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx /EPS_DECAY)
[epsilon_by_epsiode(i) for i in range(1000)]

def stack_frames(frames, state, is_new = False):
    frame = preprocess_frame(state, (104, 177, 0, 239), 84)
    frames = stack_frame(frames, frame, is_new)
    return frames

def train(n_episodes):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
    """
    env.viewer = None
    env.render()
    for i_episode in range(start_epoch + 1, n_episodes + 1):
        state = stack_frames(None, env.reset(), True)
        score = 0.0
        if i_episode < FINAL_EXPLORATION :
            eps = EPS_START
        else:
            eps = epsilon_by_epsiode(i_episode-FINAL_EXPLORATION)

        timestamp = 0

        while timestamp < 10000:
            env.render()
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(possible_actions[action])
            score += reward

            timestamp += 1

            if timestamp > 1:
                if(prev_state['health'] > info['health']):
                    #reward += (prev_state['health']-info['health'])*(-50)
                    reward += -100
                    #print(reward)

            prev_state = info

            next_state = stack_frames(state, next_state, False)
            agent.step(state, action, reward, next_state, done)
            state = next_state

            if done:
                break
        scores_window.append(score)  # save most recent score
        reward_window.append(reward)

        scores.append(score)  # save most recent score

        #print('\rEpisode {}/{}\t\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, n_episodes, np.mean(scores_window), eps), end="")
        print('Episode {}/{}\t\tScore: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, n_episodes, score, eps))

        # save model and score table every 100 episode
        if i_episode % 100 == 0:
            # save both model
            policy_path = MODEL_DIR[0]+'_{}.pth'.format(i_episode)
            target_path = MODEL_DIR[1]+'_{}.pth'.format(i_episode)
            torch.save(agent.policy_net.state_dict(), policy_path)
            torch.save(agent.target_net.state_dict(), target_path)

            df = pd.DataFrame(scores)
            df.to_csv('./scores/{}-dqn-score-{}.csv'.format(GAME_NAME,TRIAL_NUMBER))

    return scores

def deploy(n_episodes):
    env.viewer = None
    for i_episode in range(start_epoch + 1, n_episodes + 1):
        state = stack_frames(None, env.reset(), True)
        score = 0.0
        eps = 0.3
        timestamp = 0

        while timestamp < 10000:
            env.render()
            action = agent.deploy(state, eps)
            next_state, reward, done, info = env.step(possible_actions[action])
            score += reward

            timestamp += 1
            if done:
                break
        scores_window.append(score)  # save most recent score

    scores.append(score)  # save most recent score

    print('Episode {}\t\tScore {}\tAverage Score: {:.2f}'.format(n_episodes, score, np.mean(scores_window)))

#if __name__ == '__main__' :
st = time.time()

if TRAIN:
    print("Start training...")
    train(n_episodes=2000)

else:
    print('Deploying agent...')
    deploy(n_episodes=3)

elapsed_time = time.time() - st
print("")
print('Training finish')
print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
