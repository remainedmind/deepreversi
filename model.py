import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

from game import build_reversi_board, is_valid_move, apply_move

# Constants
EMPTY, BLACK, WHITE = 0, 1, -1
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

# Hyperparameters
GAMMA = 0.99
LR = 0.005
BATCH_SIZE = 128
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Current device is: {device}")

class ReversiDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ReversiDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ReversiEnv:
    def __init__(self, size=8):
        self.size = size
        self.board = build_reversi_board(size)
        self.current_player = BLACK

    def reset(self):
        self.board = build_reversi_board(self.size)
        self.current_player = BLACK
        return self.get_state()

    def step(self, action):
        row, col = divmod(action, self.size)
        if is_valid_move(self.board, row, col, self.current_player):
            apply_move(self.board, row, col, self.current_player)
            self.switch_player()
            reward = self.get_reward()
            done = self.is_done()
            next_state = self.get_state()
            return next_state, reward, done
        else:
            return self.get_state(), -1, False

    def get_state(self):
        return np.stack((self.board == BLACK, self.board == WHITE, self.board == EMPTY)).astype(np.float32)

    def get_valid_actions(self):
        """
        In that function we use `i = row * S + col` transformation to flatten a 2D board (SxS grid) into a 1D list
        :return:
        """
        return [
            r * self.size + c for r in range(self.size) for c in range(self.size) if is_valid_move(self.board, r, c, self.current_player)
        ]

    def switch_player(self):
        self.current_player *= -1

    def get_reward(self):
        black_count = np.sum(self.board == BLACK)
        white_count = np.sum(self.board == WHITE)
        return black_count - white_count

    def is_done(self):
        return not any(is_valid_move(self.board, r, c, BLACK) for r in range(self.size) for c in range(self.size)) and \
               not any(is_valid_move(self.board, r, c, WHITE) for r in range(self.size) for c in range(self.size))


@torch.inference_mode()
def model_inference(model: ReversiDQN, state: np.ndarray, actions: list[int], grid_size: int = 8):
    q_values = model(torch.tensor(state, device=device).unsqueeze(0))
    q_values = q_values.flatten()
    q_values[torch.tensor([i for i in range(grid_size * grid_size) if i not in actions])] = -float('inf')
    chosen_action = q_values.argmax().item()
    return chosen_action


def train_dqn(env: ReversiEnv, opponent_net: ReversiDQN, policy_net: ReversiDQN, epochs: int = 50, ):
    """ """
    training_player = 1
    target_net = ReversiDQN(3, env.size * env.size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.to(device=device)
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)

    epsilon = EPSILON_START
    for episode in range(epochs):
        state = env.reset()
        total_reward = 0
        while True:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            if env.current_player == training_player:
                if random.random() < epsilon:
                    action = random.choice(valid_actions)
                else:
                    action = model_inference(policy_net, state, valid_actions, env.size)
            else:
                action = model_inference(opponent_net, state, valid_actions, env.size)

            next_state, reward, done = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
                batch_state = torch.tensor(np.array(batch_state), device=device)
                batch_action = torch.tensor(np.array(batch_action), device=device).unsqueeze(1)
                batch_reward = torch.tensor(np.array(batch_reward), device=device).unsqueeze(1)
                batch_next_state = torch.tensor(np.array(batch_next_state), device=device)
                batch_done = torch.tensor(np.array(batch_done), device=device).unsqueeze(1).float()

                q_values = policy_net(batch_state).gather(1, batch_action)
                next_q_values = target_net(batch_next_state).max(1)[0].unsqueeze(1)
                target_q_values = batch_reward + GAMMA * next_q_values * (1 - batch_done)

                loss = nn.functional.mse_loss(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()




        epsilon = max(EPSILON_END, EPSILON_DECAY * epsilon)
        """
        but once the batch is full, then (after back propagation) we switch the user, 
        and the agent starts to learn as another player.
        """
        training_player *= -1  # Switch the player after a session of games

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}, Total Reward: {total_reward}")

    return policy_net


def main():
    env = ReversiEnv()
    desk_size = env.size
    opponent_model = ReversiDQN(3, desk_size * desk_size)
    policy_model = ReversiDQN(3, desk_size * desk_size)
    opponent_model.to(device=device)
    policy_model.to(device=device)
    try:
        opponent_model.load_state_dict(torch.load("reversi_policy_net.pth"))
        policy_model.load_state_dict(torch.load("reversi_policy_net.pth"))
    except FileNotFoundError:
        pass
    opponent_model.eval()

    policy_net = train_dqn(env, opponent_net=opponent_model, policy_net=policy_model, epochs=100)
    torch.save(policy_net.state_dict(), "reversi_policy_net.pth")


if __name__ == "__main__":
    main()
