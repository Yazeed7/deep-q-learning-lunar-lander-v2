import optuna
import matplotlib.pyplot as plt

from DQN import *

def train_test_best_parameters():
    print('Starting best hyperparameters experiment')

    # Train agent
    Agent = DQNAgent(2,
             64,
             0.99,
             10_000,
             1_000,
             1.0,
             0.995,
             0.01,
             0.0005,
             64,
            )
    rewards_train = Agent.train(750)

    # Plot training rewards
    plt.figure()
    n = 10 # moving average length
    moving_avg = np.convolve(rewards_train, np.ones(n)/n, mode='valid')
    plt.plot(np.arange(len(rewards_train))+1, rewards_train, label='Training reward')
    plt.plot(np.arange(len(rewards_train)-(n-1))+n, moving_avg, label=f'{n} Episode moving average')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training rewards for ideal hyperparameters')
    # plt.axhline(y=200, label='Goal reward')
    plt.legend()
    plt.savefig('RL_P2_Training.png')

    # Plot testing results
    rewards_test, _ = Agent.test(100, return_mean=False)

    plt.figure()
    plt.hist(rewards_test)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('DQN test results')
    plt.axvline(x=200, c='b', ls='--', label='Target reward')
    plt.axvline(x=rewards_test.mean(), c='r', ls=':', label='Average reward')
    plt.legend()
    plt.savefig('RL_P2_Testing.png')

def gamma_experiment():
    print('Starting gamma experiment')

    # train and test agents
    rewards_gamma = []
    ep_lens = []
    gammas = [0.9, 0.99, 0.999, 1]
    for gamma in gammas:
        Agent = DQNAgent(2,
                     64,
                     gamma,
                     10_000,
                     1_000,
                     1.0,
                     0.995,
                     0.01,
                     0.0005,
                     64,
                    )
        Agent.train(750)
        r, l = Agent.test(100)
        rewards_gamma.append(r)
        ep_lens.append(l)

    # Plot rewards
    plt.figure()
    gamma_labels = [fr'γ = {g}' for g in gammas]
    plt.bar(gamma_labels, rewards_gamma)
    plt.ylabel('Average testing reward')
    plt.title(r'Hyperparameter tuning: discount rate $.png$')
    plt.axhline(y=200, c='r', ls='--')
    plt.axhline(y=0)
    plt.savefig('RL_P2_HP_gamma.png')

    # Plot frames
    plt.figure()
    gamma_labels = [fr'γ = {g}' for g in gammas]
    plt.bar(gamma_labels, ep_lens)
    plt.ylabel('Average testing episode length')
    plt.title(r'Hyperparameter tuning: discount rate γ')
    plt.savefig('RL_P2_HP_gamma2.png')

def beta_experiment():
    print('Starting beta experiment')

    # Train and test agents
    rewards_beta = []
    betas = [0.8, 0.95, 0.995, 0.9995]
    for beta in betas:
        Agent = DQNAgent(2,
                     64,
                     0.99,
                     10_000,
                     1_000,
                     1.0,
                     beta,
                     0.01,
                     0.0005,
                     64,
                    )
        Agent.train(750)
        rewards_beta.append(Agent.test(100)[0])

    # Plot rewards
    plt.figure()
    beta_labels = [fr'β = {b}' for b in betas]
    plt.bar(beta_labels, rewards_beta)
    plt.ylabel('Average testing reward')
    plt.title(r'Hyperparameter tuning: ε decay rate β')
    plt.axhline(y=200, c='r', ls='--')
    plt.axhline(y=0)
    plt.savefig('RL_P2_HP_beta.png')

def alpha_experiment():
    print('Starting alpha experiment')

    # Train and test agents
    alphas = [5e-2, 5e-3, 5e-4, 5e-5]
    rewards_alphas = []
    for alpha in alphas:
        Agent = DQNAgent(2,
                     64,
                     0.99,
                     10_000,
                     1_000,
                     1.0,
                     0.995,
                     0.01,
                     alpha,
                     64,
                    )
        Agent.train(750)
        rewards_alphas.append(Agent.test(100)[0])

    # Plot rewards
    plt.figure()
    alpha_labels = [fr'α = {a}' for a in alphas]
    plt.bar(alpha_labels, rewards_alphas)
    plt.ylabel('Average testing reward')
    plt.title(r'Hyperparameter tuning: learning rate α')
    plt.axhline(y=200, c='r', ls='--')
    plt.axhline(y=0)
    plt.savefig('RL_P2_HP_alpha.png')

if __name__ == '__main__':
    # Need to define an agent that we ignore for seeds to work, for some reason?
    Agent = DQNAgent(2,64,0.99, 10_000,1_000,1.0,0.995,0.01,0.0005,64)
    train_test_best_parameters()
    gamma_experiment()
    beta_experiment()
    alpha_experiment()
