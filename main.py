import gym
import os
import argparse
import datetime
import time
import numpy as np
from pg_torch import PolicyGradientAgent
from utils import plot_learning_curve

seed = 1 # 1, 3
np.random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-lr', type=float, default=5e-4) 
    parser.add_argument('-fc1', type=int, default=128) 
    parser.add_argument('-fc2', type=int, default=128)
    args = parser.parse_args()
    e = datetime.datetime.now()
    
    if not os.path.exists("./plots/"):
        os.makedirs("./plots")
    
    env = gym.make('LunarLander-v2')
    n_games = 4000
    
    lr = args.lr
    fc1_dims = args.fc1
    fc2_dims = args.fc2

    agent = PolicyGradientAgent(lr = lr, input_dims=[8], gamma=0.99, n_actions=4, fc1_dims=fc1_dims, fc2_dims=fc2_dims)

    fname = 'pg_' + 'lunar_lander_lr' + str(agent.lr) + '_' \
            + str(n_games) + 'games' + '_' + str(agent.fc1_dims) + str(agent.fc2_dims)
    
    best_score = env.reward_range[0]
    start = time.time()
   
    score_history = []
    learn_iters = 0
    for i in range(n_games):
        env.seed(seed)
        env.action_space.seed(seed)
        observation = env.reset()

        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            learn_iters += 1
            score += reward
            agent.store_rewards(reward)
            observation = observation_
        agent.learn()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > 200:
            print("Enviroment solved")
            break
        
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'learning_steps', learn_iters)
    end = time.time()
    x = [i+1 for i in range(len(score_history))]
    figure_file = 'plots/' + fname + f"_time_taken_{int(end - start)}" + '.png'
    plot_learning_curve(x, score_history, figure_file)


