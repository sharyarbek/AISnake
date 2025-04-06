import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGameAIMode, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot


'''
# case №1
try: +-1000 
Record 71
BS = 1000
LR = 0.001
gamma = 0.9
linear_qnet = 256
epsilon = 200

# case №2
try: 936 
Record 86
BS = 1500
LR = 0.0003
gamma = 0.95
linear_qnet = 512
epsilon = 250

# case №3
try: 815
Record: 71
BS = 2000
LR = 0.0001
gamma = 0.99
lq = 512
epsilon = 300
'''

MAX_MEMORY = 100_000
BATCH_SIZE = 2000 # 1000 -> 1500 -> 2000
LR = 0.0001 # 0.001 -> 0.0003 -> 0.0001


# Определяем устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Для проверки

class Agent:
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.99 # discount rate  # 0.9 -> 0.95 -> 0.99
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 512, 3).to(device) # 256 -> 512 -> 512
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        #model, trainer


    def get_state(self, game):
        
        # обводка точек вокруг головы змеи
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # направление змея
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN


        state = [ # 11 states
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            
            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            
            
            # Move direction 
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            
            # Food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y, # food down
        ]
        return np.array(state, dtype=int) # 11 states
        

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        # 1st way
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        states = torch.tensor(states, dtype=torch.float).to(device) #
        actions = torch.tensor(actions, dtype=torch.long).to(device) #
        rewards = torch.tensor(rewards, dtype=torch.float).to(device) #
        next_states = torch.tensor(next_states, dtype=torch.float).to(device) #
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # 2nd way
        # for state, action, reward, next_state, done in mini_sample:
        #     self.trainer.train_step(state, action, reward, next_state, done)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(
            torch.tensor(state, dtype=torch.float).to(device),
            torch.tensor(action, dtype=torch.long).to(device),
            torch.tensor(reward, dtype=torch.float).to(device),
            torch.tensor(next_state, dtype=torch.float).to(device),
            done
        )
    
    '''def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)'''

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 300) < self.epsilon: # 200 -> 250 -> 300
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAIMode()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory 
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            
            print('Game', agent.n_games, 'Score', score, 'Record', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()