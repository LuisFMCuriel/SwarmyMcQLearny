from strategies import EGreedyExpStrategy, GreedyStrategy
from algorithms import DDQN


my_DDQN = DDQN(gamma = 1,
               env_name = 'CartPole-v1',
               update_target_every_n_steps = 4,
               training_strategy_fn = EGreedyExpStrategy(),
               evaluation_strategy_fn = GreedyStrategy(),
               lr = 0.005,
               replay_buffer_size = 100000,
               replay_buffer_batch_size = 64
               )

_, _ = my_DDQN.train(max_episodes = 1000)
my_DDQN.display_gif(filename = r"./media/CartPole-v1_pytorch.gif")