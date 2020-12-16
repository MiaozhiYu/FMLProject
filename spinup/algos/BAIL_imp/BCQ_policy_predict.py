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
			  batch_size=100, stop_crt="Q_dist", save_freq=int(1e5), max_timesteps=float(1e6) - 1,
			  logger_kwargs=dict(), plot=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     device = torch.device('cpu')
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
        print("---------------------------------")
        print("loading buffer {b}........".format(b=buffer_name))
        print("---------------------------------")
        replay_buffer.load(buffer_name)
        buffer_name += '_1000K'
        #setting_name = setting_name.replace('crt', str(desire_stop_dict[env_set]))
    elif 'Final' in buffer_type or 'sigma'in buffer_type:
        replay_buffer = utils.ReplayBuffer()
        buffer_name = buffer_type.replace('env', env_set)
        print("---------------------------------")
        print("loading buffer {b}........".format(b=buffer_name))
        print("---------------------------------")
        replay_buffer.load(buffer_name)
    elif 'optimal' in buffer_type:
        buffer_name = buffer_type.replace('env', env_set)
        print("---------------------------------")
        print("loading buffer {b}........".format(b=buffer_name))
        print("---------------------------------")
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
    training_iters = 100000
    min_index = 1

    if stop_crt == "Q_dist_rate":
            dist_lst = []
            min_index = 1
            min_rate = 10000

    if stop_crt == "Q_loss":
        min_index = 1
        min_loss = -1

    if stop_crt == "Q_test":
        Q1_lst = []
        Q2_lst = []
        min_index = 1
        min_dist = -1

    if stop_crt == "VAE_loss":
        min_index = 1
        min_loss = -1

    if stop_crt == "policy_loss":
        min_index = 1
        min_loss = 0

    if stop_crt == "Q_value":
        min_index = 1
        max_loss = 0

    while training_iters < max_timesteps:
        # Load networks
        pkl_file = open('./bcq_batchpolicy/%s/%s_s%s/vars%s.pkl' % (file_name, file_name, seed, training_iters), 'rb')
        netdict = CPU_Unpickler(pkl_file).load()
        pkl_file.close()

        double_q_net = BCQ_batchpolicy.Critic(state_dim, action_dim).to(device)
        double_q_net.load_state_dict(netdict['double_q_net'])
        double_q_target_net = BCQ_batchpolicy.Critic(state_dim, action_dim).to(device)
        double_q_target_net.load_state_dict(netdict['double_q_target_net'])
        policy_net = BCQ_batchpolicy.Actor(state_dim, action_dim, max_action).to(device)
        policy_net.load_state_dict(netdict['policy_net'])
        policy_lst.append(policy_net)


        if  stop_crt == "Q_dist":
            with torch.no_grad():
                Q1_vals, Q2_vals = double_q_net(state, action)
                Eu_dist = (Q1_vals - Q2_vals).pow(2).mean()
                EU_distance.append(np.sqrt(Eu_dist.cpu()))
                if Eu_dist < min_dist:
                    min_dist = Eu_dist
                    min_index = training_iters
            print(Eu_dist)

        if  stop_crt == "Q_dist_rate":
            with torch.no_grad():
                Q1_vals, Q2_vals = double_q_target_net(state, action)
                Eu_dist = (Q1_vals - Q2_vals).pow(2).mean()
            print(Eu_dist)
            dist_lst.append(Eu_dist)
            if len(dist_lst) != 1:
                rate = dist_lst[-1] / dist_lst[-2]
                if min_index == 1 or rate <= min_rate:
                    min_rate = rate
                    min_index = training_iters

        # Return policy with smallest Q lost
        if stop_crt == "Q_loss":
            with torch.no_grad():
                Q1_vals, Q2_vals = double_q_net(state, action)
                Q1_target_vals, Q2_target_vals = double_q_target_net(state, action)
                target_q_vals = 0.75 * torch.min(Q1_target_vals, Q2_target_vals) + 0.25 * torch.min(Q1_target_vals, Q2_target_vals)
                target_q_vals = reward + done * 0.99 * target_q_vals
                Q1_loss = torch.nn.functional.mse_loss(Q1_vals, target_q_vals)
                Q2_loss = torch.nn.functional.mse_loss(Q2_vals, target_q_vals)
                Q_loss = Q1_loss + Q2_loss

            print(Q_loss)
            if min_loss < 0 or Q_loss <= min_loss:
                min_loss = Q_loss
                min_index = training_iters

        # To test distance change for each network
        if stop_crt == "Q_test":
            with torch.no_grad():
                Q1_vals, Q2_vals = double_q_net(state, action)
            Q1_lst.append(Q1_vals)
            Q2_lst.append(Q2_vals)
            if len(Q1_lst) == 1:
                dist_1 = Q1_lst[-1].pow(2).mean()
                dist_2 = Q2_lst[-1].pow(2).mean()
                min_dist = dist_1 + dist_2
            else:
                dist_1 = (Q1_lst[-1] - Q1_lst[-2]).pow(2).mean()
                dist_2 = (Q2_lst[-1] - Q2_lst[-2]).pow(2).mean()

            dist = (dist_1 + dist_2) / 2
            if min_index == 1 or dist <= min_dist:
                min_dist = dist
                min_index = training_iters
            print(dist)

        if stop_crt == "VAE_loss":
            vae_loss = 0
            for i in range(10):
                state_np, next_state_np, action, reward, done = replay_buffer.sample(100000)
                state = torch.FloatTensor(state_np).to(device)
                action = torch.FloatTensor(action).to(device)
                vae_net = BCQ_batchpolicy.VAE(state_dim, action_dim, latent_dim, max_action).to(device)
                vae_net.load_state_dict(netdict['vae_net'])
                recon, mean, std = vae_net(state, action)
                recon_loss = torch.nn.functional.mse_loss(recon, action)
                KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
                vae_loss += float(recon_loss + 0.5 * KL_loss)

            vae_loss = vae_loss / 10
            print(vae_loss)
            if min_loss < 0 or vae_loss <= min_loss:
                min_loss = vae_loss
                min_index = training_iters

        if stop_crt == "policy_loss":
            csv = ("bcq_batchpolicy/bcq_batchpolicy_{b}_{e}/bcq_batchpolicy_{b}_{e}_s{s}".format(b=b, e=e, s=seed)).replace('.', '-').lower()
            temp = pd.read_csv(csv + "/progress.txt", sep='\t')
            step = training_iters + 10000
            policy_loss = temp.loc[temp["TotalSteps"] == step, "AverageActLoss"].iloc[0]
            print(policy_loss)
            if min_index == 1 or policy_loss <= min_loss:
                min_loss = policy_loss
                min_index = training_iters

        if stop_crt == "Q_value":
            csv = ("bcq_batchpolicy/bcq_batchpolicy_{b}_{e}/bcq_batchpolicy_{b}_{e}_s{s}".format(b=b, e=e,
                                                                                                 s=seed)).replace('.',
                                                                                                                  '-').lower()
            temp = pd.read_csv(csv + "/progress.txt", sep='\t')
            step = training_iters + 10000
            q1_avg_val = temp.loc[temp["TotalSteps"] == step, "AverageQ1Vals"].iloc[0]
            q2_avg_val = temp.loc[temp["TotalSteps"] == step, "AverageQ2Vals"].iloc[0]
            q_avg_val = (q1_avg_val + q2_avg_val) / 2
            print(q_avg_val)

            if min_index == 1 or q_avg_val >= max_loss:
                max_loss = q_avg_val
                min_index = training_iters

        training_iters += save_freq

        #     VAE loss
        #     policy loss
        #     different evaluation

    if plot:
        Qvals = pd.DataFrame.from_dict({'iters': train_iters_list, 'EU_dist': EU_distance})
        print(Qvals)
        sns.lineplot(x = 'iters', y = 'EU_dist', data = Qvals)
        plt.savefig('./eu_distance.png')
    # return stop_iter + int(1e5)
    print(min_index)
    return min_index + 10000

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
    # stop_criterion = ["Q_dist", "Q_dist_rate", "Q_loss", "policy_loss", "Q_test", "Q_value"]
    # stop_criterion = ["Q_dist", "Q_dist_rate", "Q_loss", "policy_loss"]
    stop_criterion = ["VAE_loss"]
    # stop_criterion = ["Q_value"]
    # stop_criterion = ["Q_dist"]
    summary = {"stop_criterion":[], "env": [], "buffer_type":[], "mean_rank":[]};
    seed_stop_iter = {}
    seeds = [1, 2, 3]
    print('--------------------------------------------')
    print('Aggregating progress.txt together')
    # envs = ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2"]
    # envs = ["HalfCheetah-v2", "Walker2d-v2"]
    envs = ["Hopper-v2"]
    buffers = ["FinalSigma0.5_env_3_1000K", "FinalSigma0.5_env_4_1000K"]
    # buffers = ["FinalSigma0.5_env_4_1000K"]
    steps = range(int(1e5) + int(1e4), int(1e6) + int(1e5), int(1e5))
    for s in stop_criterion:
        for e in envs:
            for b in buffers:
                print("working on env {e}".format(e=e))
                df = pd.DataFrame()
                for seed in seeds:
                    csv = ("bcq_batchpolicy/bcq_batchpolicy_{b}_{e}/bcq_batchpolicy_{b}_{e}_s{s}".format(b=b, e=e, s=seed)).replace('.', '-').lower()
                    # temp = pd.read_csv(
                    #     './bcq_batchpolicy/{e}/{e}_s{i}/progress.txt'.format(e = e, i=seed), sep='\t')
                    temp = pd.read_csv(csv+"/progress.txt", sep='\t')
                    temp['seed'] = seed
                    df = pd.concat([df, temp])
                    print('Getting the stop iteration for seed {}......'.format(seed))
                    stop_iter = bcq_stop(stop_crt=s, seed=seed, buffer_type=b, env_set=e)
                    print('----------------------------------------------')
                    seed_stop_iter[seed] = stop_iter
                df.reset_index(inplace=True)
                # files.append("{e}_concat_progress.csv".format(e=e))
                # df.to_csv("{e}_concat_progress.csv".format(e=e))
                # df = pd.read_csv("{e}_concat_progress.csv")
                step_df = df.loc[df['TotalSteps'].isin(steps)]
                step_df['rank'] = step_df[['seed', 'TotalSteps', 'AverageTestEpRet']].groupby(['seed'])['AverageTestEpRet'].rank(
                    ascending=False)
                ranks = []
                # print(step_df["AverageTestEpRet", "TotalSteps", "seed", "rank"])
                for seed, iter in seed_stop_iter.items():
                    # print(seed, iter)
                    r = step_df[step_df.seed == seed][step_df.TotalSteps == iter][['rank']].values[0][0]
                    print(seed, iter, r)
                    ranks.append(r)
                rank_mean = np.mean(ranks)
                print('-------------------------------------------')
                print('rank mean for this algo is {}'.format(rank_mean))
                summary['stop_criterion'].append(s)
                summary['env'].append(e)
                summary['buffer_type'].append(b)
                summary['mean_rank'].append(rank_mean)
    summary_df = pd.DataFrame.from_dict(summary)
    summary_df.to_csv("summary.csv")
    print("The summary report is...............")
    print(summary_df)
    print('the total run time is {}s'.format(time.time() - st))


