import gym
import numpy as np
import torch
import argparse
import os
import io
# from spinup.utils.logx import EpochLogger
# from spinup.utils.run_utils import setup_logger_kwargs
# from spinup.algos.BAIL_imp import utils, BCQ_batchpolicy
import utils, bail_training_batchpolicy
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def bail_stop(env_set="Hopper-v2", seed=1, buffer_type="FinalSigma0.5_env_3_1000K",
             batch_size=100, stop_crt="Q_dist", save_freq=int(1e4), max_timesteps=float(1e5) - 1,
             logger_kwargs=dict(), plot=False, para=2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     device = torch.device('cpu')
    print("running on device:", device)

    file_name = ("bail_batchpolicy_%s_%s" % (buffer_type, env_set)).replace('.', '-').lower()

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
        # setting_name = setting_name.replace('crt', str(desire_stop_dict[env_set]))
    elif 'Final' in buffer_type or 'sigma' in buffer_type:
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

    # state_np, next_state_np, action, reward, done = replay_buffer.sample(replay_buffer.get_length())
    # state = torch.FloatTensor(state_np).to(device)
    # action = torch.FloatTensor(action).to(device)
    # next_state = torch.FloatTensor(next_state_np).to(device)
    # reward = torch.FloatTensor(reward).to(device)
    # done = torch.FloatTensor(1 - done).to(device)

    policy_lst = []
    train_iters_list = []
    EU_distance = []
    min_dist = np.inf
    stop_iter = 0
    training_iters = 10000

    index = 1
    min_val = 0

    while training_iters < max_timesteps:
        # Load networks
        # pkl_file = open('./bail_batchpolicy/%s/%s_s%s/vars%s.pkl' % (file_name, file_name, seed, training_iters), 'rb')
        # netdict = CPU_Unpickler(pkl_file).load()
        # pkl_file.close()

        if stop_crt == "Clone_loss":
            csv = ("bail_batchpolicy/bail_batchpolicy_{b}_{e}/bail_batchpolicy_{b}_{e}_s{s}".format(b=buffer_type, e=env_set,
                                                                                                    s=seed)).replace(
                '.', '-').lower()
            temp = pd.read_csv(csv + "/progress.txt", sep='\t')
            step = training_iters

            clone_loss = temp.loc[temp["TotalSteps"] == step, "CloneLoss"].iloc[0]
            print(clone_loss)
            if index == 1 or clone_loss < min_val:
                min_val = clone_loss
                index = training_iters

        if stop_crt == "UE_loss":
            csv = ("bail_batchpolicy/bail_batchpolicy_{b}_{e}/bail_batchpolicy_{b}_{e}_s{s}".format(b=buffer_type, e=env_set,
                                                                                                    s=seed)).replace(
                '.', '-').lower()
            temp = pd.read_csv(csv + "/progress.txt", sep='\t')
            step = training_iters

            clone_loss = temp.loc[temp["TotalSteps"] == step, "UELoss"].iloc[0]
            print(clone_loss)
            if index == 1 or clone_loss < min_val:
                min_val = clone_loss
                index = training_iters

        if stop_crt == "UEValiLossMin":
            csv = ("bail_batchpolicy/bail_batchpolicy_{b}_{e}/bail_batchpolicy_{b}_{e}_s{s}".format(b=buffer_type,
                                                                                                    e=env_set,
                                                                                                    s=seed)).replace(
                '.', '-').lower()
            temp = pd.read_csv(csv + "/progress.txt", sep='\t')
            step = training_iters

            # v1 = temp.loc[temp["TotalSteps"] == step, "UEValiLossMin"].iloc[0]
            # print("v1 = {v}".format(v=v1))
            # v0 = temp.loc[temp["TotalSteps"] == step - 1000, "UEValiLossMin"].iloc[0]
            # print("v0 = {v}".format(v=v0))
            # v2 = temp.loc[temp["TotalSteps"] == step + 1000, "UEValiLossMin"].iloc[0]
            # print("v2 = {v}".format(v=v2))

            val = 0

            for i in range(-1 * para, 0):
                u = temp.loc[temp["TotalSteps"] == step + 1000 * i, "UEValiLossMin"].iloc[0]
                v = temp.loc[temp["TotalSteps"] == step + 1000 * (i + 1), "UEValiLossMin"].iloc[0]
                print("[{i}, {j}]: {u}  {v}".format(i=i, j=i+1, u=u, v=v))
                val += np.abs(v - u)

            val = val / para

            print(val)

            if index == 1 or val < min_val:
                min_val = val
                index = training_iters


        training_iters += save_freq

        #     VAE loss
        #     policy loss
        #     different evaluation

    if plot:
        Qvals = pd.DataFrame.from_dict({'iters': train_iters_list, 'EU_dist': EU_distance})
        print(Qvals)
        sns.lineplot(x='iters', y='EU_dist', data=Qvals)
        plt.savefig('./eu_distance.png')
    # return stop_iter + int(1e5)
    print(index)
    return index


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

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":
    st = time.time()

    # stop_criterion = ["Q_dist", "Q_dist_rate", "Q_loss", "policy_loss", "Q_test"]
    # stop_criterion = ["VAE_loss"]
    # stop_criterion = ["Q_value"]

    # stop_criterion = ["Clone_loss", "UE_loss", "UEValiLossMin"]
    stop_criterion = ["UEValiLossMin"]

    summary = {"stop_criterion": [], "env": [], "buffer_type": [], "mean_rank": []};
    seed_stop_iter = {}
    seeds = [1, 2, 3]
    print('--------------------------------------------')
    print('Aggregating progress.txt together')
    envs = ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2"]
    # envs = ["Walker2d-v2"]
    buffers = ["FinalSigma0.5_env_3_1000K", "FinalSigma0.5_env_4_1000K"]
    # buffers = ["FinalSigma0.5_env_4_1000K"]
    steps = range(int(1e4), int(1e5), int(1e4))
    bail_stop()
    for s in stop_criterion:
        for e in envs:
            for b in buffers:
                print("working on env {e}".format(e=e))
                df = pd.DataFrame()
                for seed in seeds:
                    csv = ("bail_batchpolicy/bail_batchpolicy_{b}_{e}/bail_batchpolicy_{b}_{e}_s{s}".format(b=b, e=e,
                                                                                                         s=seed)).replace(
                        '.', '-').lower()

                    temp = pd.read_csv(csv + "/progress.txt", sep='\t')
                    temp['seed'] = seed
                    df = pd.concat([df, temp])
                    print('Getting the stop iteration for seed {}......'.format(seed))
                    stop_iter = bail_stop(stop_crt=s, seed=seed, buffer_type=b, env_set=e, para=5)
                    print('----------------------------------------------')
                    seed_stop_iter[seed] = stop_iter
                df.reset_index(inplace=True)
                # files.append("{e}_concat_progress.csv".format(e=e))
                # df.to_csv("{e}_concat_progress.csv".format(e=e))
                # df = pd.read_csv("{e}_concat_progress.csv")
                step_df = df.loc[df['TotalSteps'].isin(steps)]
                step_df['rank'] = step_df[['seed', 'TotalSteps', 'AverageTestEpRet']].groupby(['seed'])[
                    'AverageTestEpRet'].rank(
                    ascending=False)
                ranks = []
                print(step_df)
                for seed, iter in seed_stop_iter.items():
                    # print(seed, iter)
                    # print(step_df.loc[((step_df["TotalSteps"] == iter) & (step_df["seed"] == seed)), "rank"].values[0])
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
    summary_df.to_csv("bail_summary.csv")
    print("The summary report is...............")
    print(summary_df)
    print('the total run time is {}s'.format(time.time() - st))


