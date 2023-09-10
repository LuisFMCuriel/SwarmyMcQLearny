from strategies import EGreedyExpStrategy, GreedyStrategy
from algorithms import DDQN
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='DDQN Training Script')
    parser.add_argument('--gamma', type=float, default=1, help='Discount factor (gamma)')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Name of the environment')
    parser.add_argument('--update_target_every_n_steps', type=int, default=4, help='Update target network every n steps')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--replay_buffer_size', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--replay_buffer_batch_size', type=int, default=64, help='Replay buffer batch size')
    parser.add_argument('--max_episodes', type=int, default=1000, help='Maximum number of episodes')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()

    my_DDQN = DDQN(
        gamma=args.gamma,
        env_name=args.env_name,
        update_target_every_n_steps=args.update_target_every_n_steps,
        training_strategy_fn=EGreedyExpStrategy(),
        evaluation_strategy_fn=GreedyStrategy(),
        lr=args.lr,
        replay_buffer_size=args.replay_buffer_size,
        replay_buffer_batch_size=args.replay_buffer_batch_size
    )

    _, _ = my_DDQN.train(max_episodes=args.max_episodes)
    my_DDQN.display_gif(filename=r"./media/CartPole-v1_pytorch.gif")