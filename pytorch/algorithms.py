from itertools import count
import torch
from models import FCQ
import numpy as np
from IPython.display import display, Image
from strategies import GreedyStrategy
from replayBuffer import ReplayBuffer
import PIL
import time
import gym
import torch.optim as optim


class DDQN:

    def __init__(self,
                 gamma: float,
                 training_strategy_fn,
                 evaluation_strategy_fn,
                 lr: float = 0.005,
                 env_name: str = 'CartPole-v1',
                 update_target_every_n_steps: int = 4,
                 replay_buffer_size = 100000,
                 replay_buffer_batch_size = 64):

        self.gamma = gamma
        self.env_name = env_name
        self.update_target_every_n_steps = update_target_every_n_steps
        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.lr = lr
        self.replay_buffer = ReplayBuffer(max_size = replay_buffer_size, batch_size = replay_buffer_batch_size)


    def update_network(self, tau):
        for target, online in zip(self.target_model.parameters(), self.online_model.parameters()):
            target.data.copy_(tau*online.data + (1.0 - tau)*target.data)


    def optimize_model(self,
                       experiences,
                       max_gradient_norm = float('inf')):
        global q_sp
        global max_a_q_sp
        global argmax_a_q_sp
        global target_q_sa
        global q_sa
        global states
        global actions

        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = len(is_terminals)
        # We get the argmax (or maximum action index using the online network)
        argmax_a_q_sp = self.online_model(next_states).max(1)[1]
        # Then we use the target model to calculate the estimated Q-values
        q_sp = self.target_model(next_states).detach()
        # And extract the max value using the index gotten with the online model
        max_a_q_sp = q_sp[np.arange(batch_size), argmax_a_q_sp].unsqueeze(1)

        # Then we start computing the loss value
        target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))
        q_sa = self.online_model(states).gather(1, actions)

        td_error = q_sa - target_q_sa
        value_loss = td_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), max_gradient_norm)
        self.value_optimizer.step()

    def evaluate(self, eval_policy_model, eval_env, n_episodes=1):
        rs = []
        evaluation_strategy = self.evaluation_strategy_fn
        for _ in range(n_episodes):
            s, d = eval_env.reset(), False
            rs.append(0)
            for _ in count():
                a = evaluation_strategy.select_action(eval_policy_model, s)
                s, r, d, _ = eval_env.step(a)
                rs[-1] += r
                if d: break
        return np.mean(rs), np.std(rs)


    def train(self,
                   hidden_dims = (512, 128),
                   batch_size = 64,
                   n_warmup_batches = 5,
                   goal_mean = 475,
                   max_episodes = 1000,
                   tau = 1):


        env = gym.make(self.env_name)

        nS, nA = env.observation_space.shape[0], env.action_space.n
        self.target_model = FCQ(nS, nA, hidden_dims=hidden_dims)
        self.online_model = FCQ(nS, nA, hidden_dims=hidden_dims)
        self.best_model = FCQ(nS, nA, hidden_dims=hidden_dims)
        best_score = 0
        self.update_network(tau = tau)
        self.value_optimizer = optim.RMSprop(self.online_model.parameters(), lr = self.lr)
        min_samples = batch_size*n_warmup_batches


        #max_gradient_norm = float('inf') # For backpropagation
        n_network_updates = 0
        # Lists to store training details
        episode_reward = []
        episode_seconds = []
        episode_timestep = []
        evaluation_scores = []
        mean_reward_10_episodes_arr = []
        episodes_arr = []
        mean_100_eval_score_arr = []
        # 7.- Start training
        # Start the timer
        training_start = time.time()
        for episode in range(max_episodes + 1):
            # Let's add some timing to see how the agent develops. For this specific environment, the more time the agent spends in the episode better (it kept the pole in the right position)
            episode_start = time.time()
            # Reset the environment
            state = env.reset()

            #state = state[0]
            is_terminal = False

            # Start storing data for the whole episode
            episode_reward.append(0.0)
            episode_timestep.append(0.0)
            for timestep, step in enumerate(count()):
                # First the agent selects an action (the online model)
                action = self.training_strategy_fn.select_action(self.online_model, state)
                # Make the action and get the infor of the next state
                new_state, reward, is_terminal, info = env.step(action)
                # Update the data
                episode_reward[-1] += reward
                episode_timestep[-1] += 1
                # Check if we reach the maximum number of steps for the environment or if we lost the game
                is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
                is_failure = is_terminal and not is_truncated
                # Save the info for the replay buffer
                experience = (state, action, reward, new_state, float(is_failure))
                self.replay_buffer.store(experience)
                state = new_state
                if len(self.replay_buffer) > min_samples:
                    # Load stored data
                    experiences = self.replay_buffer.sample()
                    # Transform the experiences in tensors
                    experiences = self.online_model.load(experiences)
                    self.optimize_model(experiences)

                # Check if the game is over
                if is_terminal:
                    break
                # Copy the online weights into the target network
                if np.sum(episode_timestep) % self.update_target_every_n_steps == 0:
                    self.update_network(tau = tau)
                    n_network_updates += 1


            evaluation_score, std_eval = self.evaluate(eval_policy_model = self.online_model, eval_env = env)
            evaluation_scores.append(evaluation_score)
            mean_100_eval_score = np.mean(evaluation_scores[-100:])

            if evaluation_score > best_score:
                self.best_model = self.online_model
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            mean_reward_10_episodes = np.mean(episode_reward[-10:])
            mean_reward_10_episodes_arr.append(mean_reward_10_episodes)
            mean_100_eval_score_arr.append(mean_100_eval_score)
            episodes_arr.append(episode)
            message = "elapsed time: {}, episode: {}, episode timestep: {}, reward 10 episodes: {}, eval mean: {}".format(elapsed_str, episode + 1, timestep, mean_reward_10_episodes, mean_100_eval_score)
            print(message)
            if mean_100_eval_score >= goal_mean:
                print("Reached goal -> training complete")
                break
        return episodes_arr, mean_100_eval_score_arr

    def create_gif(self, episodes=5, max_steps=500):
        env = gym.make(self.env_name)
        agent = self.best_model
        frames = []

        for _ in range(episodes):
            state = env.reset()
            done = False
            Strategy = GreedyStrategy()

            for _ in range(max_steps):
                frames.append(env.render(mode='rgb_array'))
                action = Strategy.select_action(agent, state)
                state, _, done, _ = env.step(action)

                if done:
                    break

        env.close()

        # Save frames as a GIF using imageio
        PIL_frames = [PIL.Image.fromarray(frame) for frame in frames]
        PIL_frames[0].save(self.env_name + ".gif", format='GIF', append_images=PIL_frames[1:], save_all=True, duration=1000/30, loop=0)
        #imageio.mimsave(self.env_name, frames, fps=30)

    def display_gif(self, filename: str = ".", episodes: int = 5, max_steps: int = 500):
        self.create_gif(episodes, max_steps)
        display(Image(filename=r"/content/" + self.env_name + ".gif", format='png'))
