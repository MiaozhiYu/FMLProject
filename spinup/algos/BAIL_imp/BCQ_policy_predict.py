import gym
import numpy as np
import torch
import argparse
import os
import io
#from spinup.utils.logx import EpochLogger
#from spinup.utils.run_utils import setup_logger_kwargs
#from spinup.algos.BAIL_imp import utils, BCQ_batchpolicy
import utils, BCQ_batchpolicy
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)
        
def bcq_stop(env_set="Hopper-v2", seed=1, buffer_type="FinalSigma0.5_env_3_1000K",
			  batch_size=100, stop_crt="Q_dist", save_freq=int(1e5), max_timesteps=float(1e6),
			  logger_kwargs=dict(), plot=False):

#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    print("running on device:", device)

    file_name = ("bcq_batchpolicy_%s_%s" % (buffer_type, env_set)).replace('.', '-').lower()

    print("---------------------------------------")
    print
    ("Task: " + file_name)
    print("Stop criterion:", stop_crt)
    print("---------------------------------------")


    # get env info
    env = gym.make(env_set)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # get other training info
    latent_dim = action_dim * 2

    # Load buffer
    if 'sac' in buffer_type:
        replay_buffer = utils.BEAR_ReplayBuffer()
        desire_stop_dict = {'Hopper-v2': 1000, 'Walker2d-v2': 500, 'HalfCheetah-v2': 4000, 'Ant-v2': 750}
        buffer_name = buffer_type.replace('env', env_set).replace('crt', str(desire_stop_dict[env_set]))
        replay_buffer.load(buffer_name)
        buffer_name += '_1000K'
        #setting_name = setting_name.replace('crt', str(desire_stop_dict[env_set]))
    elif 'Final' in buffer_type or 'sigma'in buffer_type:
        replay_buffer = utils.ReplayBuffer()
        buffer_name = buffer_type.replace('env', env_set)
        replay_buffer.load(buffer_name)
    elif 'optimal' in buffer_type:
        buffer_name = buffer_type.replace('env', env_set)
        replay_buffer = utils.ReplayBuffer()
        replay_buffer.load(buffer_name)
    else:
        raise FileNotFoundError('! Unknown type of dataset %s' % buffer_type)

    # Uniformly sample a size 100 batch from the batch
    # state_np, next_state_np, action, reward, done = replay_buffer.sample(100)
    state_np, next_state_np, action, reward, done =replay_buffer.sample(replay_buffer.get_length())
    state = torch.FloatTensor(state_np).to(device)
    action = torch.FloatTensor(action).to(device)
    next_state = torch.FloatTensor(next_state_np).to(device)
    reward = torch.FloatTensor(reward).to(device)
    done = torch.FloatTensor(1 - done).to(device)

    policy_lst = []
    train_iters_list = []
    EU_distance = []
    min_dist = np.inf
    stop_iter = 0
    training_iters = 0
    while training_iters < max_timesteps:
        # Load networks
        pkl_file = open('%s/%s_s%s/vars%s.pkl' % (file_name, file_name, seed, training_iters), 'rb')
        netdict = CPU_Unpickler(pkl_file).load()
        pkl_file.close()

        double_q_net = BCQ_batchpolicy.Critic(state_dim, action_dim).to(device)
        double_q_net.load_state_dict(netdict['double_q_net'])
        policy_net = BCQ_batchpolicy.Actor(state_dim, action_dim, max_action).to(device)
        policy_net.load_state_dict(netdict['policy_net'])
        policy_lst.append(policy_net)
        vae_net = BCQ_batchpolicy.VAE(state_dim, action_dim, latent_dim, max_action).to(device)
        vae_net.load_state_dict(netdict['vae_net'])

        # calculate statistics
        if  stop_crt == "Q_dist":
            with torch.no_grad():
                Q1_vals, Q2_vals = double_q_net(state, action)
                Eu_dist = (Q1_vals - Q2_vals).pow(2).mean()
                EU_distance.append(np.sqrt(Eu_dist))
                if Eu_dist < min_dist:
                    min_dist = Eu_dist
                    stop_iter = training_iters
            print(Eu_dist)

        if stop_crt == "Q-loss":
            with torch.no_grad():
                Q1_vals, Q2_vals = double_q_net(state, action)
                # Eu_dist = (Q1_vals - Q2_vals).pow(2).mean()
            print(Q1_vals.mean(), Q2_vals.mean())
        train_iters_list.append(training_iters)
        training_iters += save_freq

    if plot:
        Qvals = pd.DataFrame.from_dict({'iters': train_iters_list, 'EU_dist': EU_distance})
        print(Qvals)
        sns.lineplot(x = 'iters', y = 'EU_dist', data = Qvals)
        plt.savefig('./eu_distance.png')
    return stop_iter + int(1e5)

# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, env, eval_episodes=10):
	tol_reward = 0
	for _ in range(eval_episodes):
		obs = env.reset()
		done = False
		while not done:
			action = policy.select_action(np.array(obs))
			obs, reward, done, _ = env.step(action)
			tol_reward += reward

	avg_reward = tol_reward / eval_episodes

	print ("---------------------------------------")
	print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
	print ("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
    st = time.time()
    seed_stop_iter = {}
    seeds = [1, 2, 3]
    print('--------------------------------------------')
    print('Aggregating progress.txt together')
    df = pd.DataFrame()
    for seed in seeds:
        temp = pd.read_csv(
            'bcq_batchpolicy_finalsigma0-5_env_3_1000k_hopper-v2/bcq_batchpolicy_finalsigma0-5_env_3_1000k_hopper-v2_s{i}/progress.txt'.format(
                i=seed), sep='\t')
        temp['seed'] = seed
        df = pd.concat([df, temp])
        print('Getting the stop iteration for seed {}......'.format(seed))
        stop_iter = bcq_stop(seed = seed)
        print('----------------------------------------------')
        seed_stop_iter[seed] = stop_iter
    df.reset_index(inplace=True)

    steps = range(int(1e5), int(1e6) + int(1e5), int(1e5))
    step_df = df.loc[df['TotalSteps'].isin(steps)]
    step_df['rank'] = step_df[['seed', 'TotalSteps', 'AverageTestEpRet']].groupby(['seed'])['AverageTestEpRet'].rank(
        ascending=False)
    ranks = []
    for seed, iter in seed_stop_iter.items():
        print(seed, iter)
        r = step_df[step_df.seed == seed][step_df.TotalSteps == iter][['rank']].values[0][0]
        ranks.append(r)
    rank_mean = np.mean(ranks)
    print('-------------------------------------------')
    print('rank mean for this algo is {}'.format(rank_mean))
    print('the total run time is {}s'.format(time.time()-st))



