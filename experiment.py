import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_probability as tfp
import pandas as pd
import json
import os
import gzip
from datetime import datetime
import scipy.stats as stats
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm.notebook import tqdm
import time
import gc


# Define game classes
class BaseGame:
    """
    Base class for games.
    """

    def __init__(self):
        self.state_size = None
        self.action_size = None
        self.action_space = "discrete"  # Define action space as discrete

    def reset(self):
        pass

    def step(self, action_a, action_b):
        pass


class GameOfChicken(BaseGame):
    def __init__(self):
        super().__init__()
        self.state_size = 1
        self.action_size = 2

    def reset(self):
        return np.array([0]), np.array([0])

    def step(self, action_a, action_b):
        if action_a == action_b == 1:
            reward_a, reward_b, outcome = -100, -100, -1  # Both crash
        elif action_a == 0 and action_b == 1:
            reward_a, reward_b, outcome = -1, 1, 0  # A swerves, B straight
        elif action_a == 1 and action_b == 0:
            reward_a, reward_b, outcome = 1, -1, 1  # A straight, B swerves
        else:
            reward_a, reward_b, outcome = 0, 0, 0  # Both swerve

        next_state_a, next_state_b = np.array([0]), np.array([0])
        return reward_a, reward_b, outcome, next_state_a, next_state_b


class PrisonersDilemma(BaseGame):
    def __init__(self):
        super().__init__()
        self.state_size = 1
        self.action_size = 2

    def reset(self):
        return np.array([0]), np.array([0])

    def step(self, action_a, action_b):
        if action_a == 1 and action_b == 1:
            reward_a, reward_b, outcome = -2, -2, 0  # Both defect
        elif action_a == 0 and action_b == 1:
            reward_a, reward_b, outcome = -3, 0, 1  # A cooperates, B defects
        elif action_a == 1 and action_b == 0:
            reward_a, reward_b, outcome = 0, -3, 1  # A defects, B cooperates
        else:
            reward_a, reward_b, outcome = -1, -1, 0  # Both cooperate

        next_state_a, next_state_b = np.array([0]), np.array([0])
        return reward_a, reward_b, outcome, next_state_a, next_state_b


class StagHunt(BaseGame):
    def __init__(self):
        super().__init__()
        self.state_size = 1
        self.action_size = 2

    def reset(self):
        return np.array([0]), np.array([0])

    def step(self, action_a, action_b):
        if action_a == 1 and action_b == 1:
            reward_a, reward_b, outcome = 3, 3, 0  # Both hunt stag
        elif action_a == 0 and action_b == 1:
            reward_a, reward_b, outcome = 2, 0, 1  # A hunts hare, B hunts stag
        elif action_a == 1 and action_b == 0:
            reward_a, reward_b, outcome = 0, 2, 1  # A hunts stag, B hunts hare
        else:
            reward_a, reward_b, outcome = 2, 2, 0  # Both hunt hare

        next_state_a, next_state_b = np.array([0]), np.array([0])
        return reward_a, reward_b, outcome, next_state_a, next_state_b


class EntryDeterrenceGame(BaseGame):
    def __init__(self):
        super().__init__()
        self.state_size = 1
        self.action_size = 2

    def reset(self):
        return np.array([0]), np.array([0])

    def step(self, action_a, action_b):
        if action_a == 0:  # Incumbent sets low price
            reward_a, reward_b, outcome = (
                (1, -1, 0) if action_b == 1 else (0, 0, 0)
            )  # Entrant enters if B == 1, stays out if B == 0
        else:  # Incumbent sets high price
            reward_a, reward_b, outcome = (
                (0, 0, 0) if action_b == 1 else (2, -2, 0)
            )  # Entrant enters if B == 1, stays out if B == 0

        next_state_a, next_state_b = np.array([0]), np.array([0])
        return reward_a, reward_b, outcome, next_state_a, next_state_b


class UltimatumGame(BaseGame):
    def __init__(self):
        super().__init__()
        self.state_size = 1
        self.action_size = 11  # Assume proposer can offer between 0-10 units

    def reset(self):
        return np.array([0]), np.array([0])

    def step(self, action_a, action_b):
        if action_b <= action_a:
            reward_a, reward_b, outcome = 10 - action_a, action_a, 0  # Offer accepted
        else:
            reward_a, reward_b, outcome = 0, 0, 0  # Offer rejected

        next_state_a, next_state_b = np.array([0]), np.array([0])
        return reward_a, reward_b, outcome, next_state_a, next_state_b


class CoordinationGame(BaseGame):
    def __init__(self):
        super().__init__()
        self.state_size = 1
        self.action_size = 2

    def reset(self):
        return np.array([0]), np.array([0])

    def step(self, action_a, action_b):
        if action_a == action_b:
            reward_a, reward_b, outcome = 2, 2, 0  # Both coordinate
        else:
            reward_a, reward_b, outcome = 0, 0, 0  # No coordination

        next_state_a, next_state_b = np.array([0]), np.array([0])
        return reward_a, reward_b, outcome, next_state_a, next_state_b


class BertrandCompetition:
    def __init__(self):
        self.state_size = 2
        self.action_size = 1
        self.action_space = "continuous"
        self.price_range = (0, 10)
        self.cost_range = (0, 0)  # Setting costs to zero to model price directly
        self.demand = 100
        self.tolerance = 1e-2  # Tolerance level for price comparison

    def reset(self):
        cost_a = np.random.uniform(*self.cost_range)
        cost_b = np.random.uniform(*self.cost_range)
        state_a = np.array([cost_a, cost_b])
        state_b = np.array([cost_a, cost_b])
        return state_a, state_b

    def step(self, action_a, action_b, state_a=[0], state_b=[0]):
        price_a, price_b = action_a, action_b
        cost_a, cost_b = state_a[0], state_b[0]

        if abs(price_a - price_b) < self.tolerance:
            reward_a = (price_a - cost_a) * (self.demand / 2)
            reward_b = (price_b - cost_b) * (self.demand / 2)
            outcome = 0  # Prices are even
        elif price_a < price_b:
            reward_a = (price_a - cost_a) * self.demand
            reward_b = 0
            outcome = -1  # Agent A undercuts
        else:
            reward_a = 0
            reward_b = (price_b - cost_b) * self.demand
            outcome = 1  # Agent B undercuts

        next_state_a, next_state_b = self.reset()
        return reward_a, reward_b, outcome, next_state_a, next_state_b


class PublicGoodsGame(BaseGame):
    def __init__(self):
        super().__init__()
        self.state_size = 1
        self.action_size = 11  # Assume players can contribute between 0-10 units
        self.multiplier = 1.01
        self.num_players = 2

    def reset(self):
        return np.array([0]), np.array([0])

    def step(self, action_a, action_b):
        total_contribution = action_a + action_b
        total_value = total_contribution * self.multiplier
        reward_a = (total_value / self.num_players) - action_a
        reward_b = (total_value / self.num_players) - action_b
        outcome = 0

        next_state_a, next_state_b = np.array([0]), np.array([0])
        return reward_a, reward_b, outcome, next_state_a, next_state_b


# Potential New Games


class BargainingGame(BaseGame):
    def __init__(self):
        super().__init__()
        self.state_size = 1
        self.action_size = 11  # Assume bargaining offer can be between 0-10 units

    def reset(self):
        return np.array([0]), np.array([0])

    def step(self, action_a, action_b):
        if action_a + action_b >= 10:
            reward_a, reward_b, outcome = (
                action_a,
                10 - action_a,
                0,
            )  # Agreement reached
        else:
            reward_a, reward_b, outcome = 0, 0, 0  # No agreement

        next_state_a, next_state_b = np.array([0]), np.array([0])
        return reward_a, reward_b, outcome, next_state_a, next_state_b


class MatchingPennies(BaseGame):
    def __init__(self):
        super().__init__()
        self.state_size = 1
        self.action_size = 2

    def reset(self):
        return np.array([0]), np.array([0])

    def step(self, action_a, action_b):
        if action_a == action_b:
            reward_a, reward_b, outcome = 1, -1, 0  # A wins
        else:
            reward_a, reward_b, outcome = -1, 1, 0  # B wins

        next_state_a, next_state_b = np.array([0]), np.array([0])
        return reward_a, reward_b, outcome, next_state_a, next_state_b


class RockPaperScissors(BaseGame):
    def __init__(self):
        super().__init__()
        self.state_size = 1
        self.action_size = 3

    def reset(self):
        return np.array([0]), np.array([0])

    def step(self, action_a, action_b):
        if action_a == action_b:
            reward_a, reward_b, outcome = 0, 0, 0  # Draw
        elif (action_a - action_b) % 3 == 1:
            reward_a, reward_b, outcome = 1, -1, 0  # A wins
        else:
            reward_a, reward_b, outcome = -1, 1, 0  # B wins

        next_state_a, next_state_b = np.array([0]), np.array([0])
        return reward_a, reward_b, outcome, next_state_a, next_state_b


class BattleOfTheSexes(BaseGame):
    def __init__(self):
        super().__init__()
        self.state_size = 1
        self.action_size = 2

    def reset(self):
        return np.array([0]), np.array([0])

    def step(self, action_a, action_b):
        if action_a == 0 and action_b == 0:
            reward_a, reward_b, outcome = 2, 1, 0  # A's preferred activity
        elif action_a == 1 and action_b == 1:
            reward_a, reward_b, outcome = 1, 2, 0  # B's preferred activity
        else:
            reward_a, reward_b, outcome = 0, 0, 0  # Different activities

        next_state_a, next_state_b = np.array([0]), np.array([0])
        return reward_a, reward_b, outcome, next_state_a, next_state_b


# Define agent and utility functions
def convert_to_serializable(obj):
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    elif isinstance(obj, tf.Tensor):
        return obj.numpy().tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj


class BaseAgent:
    def __init__(
        self,
        state_size,
        action_size,
        action_space="discrete",
        base_lr=0.01,
        batch_size=128,
        decay_rate=0.95,
        discount_factor=0.95,
        transparency=False,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.transparency = transparency
        self.opponent_policy = None
        self.batch_size = batch_size

        # Learning rate schedule
        self.learning_rate = ExponentialDecay(
            initial_learning_rate=base_lr,  # Scaling learning rate with batch size
            decay_steps=100,
            decay_rate=decay_rate,
        )

        self.memory = []
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_losses = []
        self.critic_losses = []
        self.policy_history = []  # Track policy over time
        self.clip_value = 1.0  # Gradient clipping value

    def build_actor(self):
        state_input = Input(shape=(self.state_size,))
        policy_input_shape = (
            (self.action_size,)
            if self.transparency and self.action_space == "discrete"
            else (
                (self.state_size,)
                if self.transparency and self.action_space == "continuous"
                else (0,)
            )
        )
        policy_input = Input(shape=policy_input_shape)
        x = Concatenate()([state_input, policy_input])
        if self.action_space == "discrete":
            output = Dense(self.action_size, activation="softmax")(x)
        elif self.action_space == "continuous":
            mu = Dense(1, activation="tanh")(x)  # Mean of Gaussian
            sigma = Dense(1, activation="softplus")(x)  # Standard deviation of Gaussian
            output = [mu, sigma]
        else:
            raise ValueError(f"Unsupported action_space: {self.action_space}")
        model = Model(inputs=[state_input, policy_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def build_critic(self):
        state_input = Input(shape=(self.state_size,))
        policy_input_shape = (
            (self.action_size,)
            if self.transparency and self.action_space == "discrete"
            else (
                (self.state_size,)
                if self.transparency and self.action_space == "continuous"
                else (0,)
            )
        )
        policy_input = Input(shape=policy_input_shape)
        x = Concatenate()([state_input, policy_input])
        output = Dense(1, activation="linear")(x)
        model = Model(inputs=[state_input, policy_input], outputs=output)
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def choose_action(self, state):
        state = np.reshape(state, [1, self.state_size])
        if self.transparency and self.opponent_policy is not None:
            policy_input_shape = (
                (self.action_size,)
                if self.action_space == "discrete"
                else (self.state_size,)
            )
            policy_input = np.reshape(self.opponent_policy, [1, *policy_input_shape])
        else:
            policy_input = np.zeros((1, 0))

        if self.action_space == "discrete":
            policy = self.actor([state, policy_input]).numpy().flatten()
            policy = policy / policy.sum()  # Normalize probabilities
            self.policy_history.append(policy.tolist())  # Store policy
            action = np.random.choice(self.action_size, p=policy)
        elif self.action_space == "continuous":
            mu, sigma = self.actor([state, policy_input])
            mu, sigma = mu.numpy().flatten(), sigma.numpy().flatten()
            sigma = np.clip(sigma, 1e-3, 1.0)  # Avoid very small sigma values
            dist = tfp.distributions.Normal(loc=mu, scale=sigma)
            action = dist.sample().numpy().flatten()
            action = np.clip(action, 0, 10)  # Ensure action is within the desired range
            if np.isnan(action).any() or np.isnan(mu).any() or np.isnan(sigma).any():
                print(f"NaN detected in actor output: mu={mu}, sigma={sigma}")
            self.policy_history.append([mu.tolist(), sigma.tolist()])  # Store policy
        else:
            raise ValueError(f"Unsupported action_space: {self.action_space}")
        return action

    def store_experience(self, state, action, reward, next_state):
        if self.action_space == "continuous":
            action = action.tolist()
        self.memory.append((state.tolist(), action, float(reward), next_state.tolist()))
        if len(self.memory) >= self.batch_size:
            self.train_batch()

    def train_batch(self):
        batch = self.memory[: self.batch_size]
        self.memory = self.memory[self.batch_size :]

        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])

        if self.transparency and self.opponent_policy is not None:
            policy_input_shape = (
                (self.action_size,)
                if self.action_space == "discrete"
                else (self.state_size,)
            )
            policy_input = np.array(
                [self.opponent_policy for _ in range(self.batch_size)]
            )
        else:
            policy_input = np.zeros((self.batch_size, 0))

        policy_input = np.reshape(policy_input, (self.batch_size, -1))

        values = self.critic([states, policy_input]).numpy().flatten()
        next_values = self.critic([next_states, policy_input]).numpy().flatten()

        td_targets = rewards + self.discount_factor * next_values
        td_errors = td_targets - values

        with tf.GradientTape() as tape:
            values = self.critic([states, policy_input], training=True)
            critic_loss = tf.reduce_mean(tf.square(td_targets - values))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        clipped_critic_grads = [
            tf.clip_by_value(grad, -self.clip_value, self.clip_value)
            for grad in critic_grads
        ]
        self.critic.optimizer.apply_gradients(
            zip(clipped_critic_grads, self.critic.trainable_variables)
        )
        self.critic_losses.append(critic_loss.numpy())

        if self.action_space == "discrete":
            actions_one_hot = tf.one_hot(actions, self.action_size)
            with tf.GradientTape() as tape:
                policies = self.actor([states, policy_input], training=True)
                log_policies = tf.reduce_sum(
                    actions_one_hot * tf.math.log(policies), axis=1
                )
                actor_loss = -tf.reduce_mean(log_policies * td_errors)
        elif self.action_space == "continuous":
            with tf.GradientTape() as tape:
                mus, sigmas = self.actor([states, policy_input], training=True)
                sigmas = tf.clip_by_value(
                    sigmas, 1e-2, 1.0
                )  # Avoid very small sigma values
                dists = tfp.distributions.Normal(loc=mus, scale=sigmas)
                log_probs = dists.log_prob(actions)
                actor_loss = -tf.reduce_mean(log_probs * td_errors)
        else:
            raise ValueError(f"Unsupported action_space: {self.action_space}")

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        clipped_actor_grads = [
            tf.clip_by_value(grad, -self.clip_value, self.clip_value)
            for grad in actor_grads
        ]
        self.actor.optimizer.apply_gradients(
            zip(clipped_actor_grads, self.actor.trainable_variables)
        )
        self.actor_losses.append(actor_loss.numpy())


# Simulation function with chunking
def simulate_game(
    agent_a, agent_b, game, rounds=1000, chunk_size=10000, progress_queue=None
):
    outcomes = []
    actions_a = []
    actions_b = []
    action_distributions_a = []
    action_distributions_b = []
    policy_history_a = []
    policy_history_b = []
    state_a, state_b = game.reset()

    for round in range(rounds):
        if agent_a.transparency:
            state_a_reshaped = np.reshape(state_a, [1, agent_b.state_size])
            policy_input_zeros = np.zeros((1, 0))
            opponent_policy_b = agent_b.actor([state_a_reshaped, policy_input_zeros])
            opponent_policy_b = [p.numpy().flatten() for p in opponent_policy_b]
            agent_a.opponent_policy = opponent_policy_b

        if agent_b.transparency:
            state_b_reshaped = np.reshape(state_b, [1, agent_a.state_size])
            policy_input_zeros = np.zeros((1, 0))
            opponent_policy_a = agent_a.actor([state_b_reshaped, policy_input_zeros])
            opponent_policy_a = [p.numpy().flatten() for p in opponent_policy_a]
            agent_b.opponent_policy = opponent_policy_a

        action_a = agent_a.choose_action(state_a)
        action_b = agent_b.choose_action(state_b)

        if game.action_space == "discrete":
            reward_a, reward_b, outcome, next_state_a, next_state_b = game.step(
                action_a, action_b
            )
        else:
            reward_a, reward_b, outcome, next_state_a, next_state_b = game.step(
                action_a, action_b, state_a, state_b
            )

        outcomes.append((reward_a, reward_b))
        actions_a.append(action_a)
        actions_b.append(action_b)
        policy_history_a.append(agent_a.policy_history[-1])
        policy_history_b.append(agent_b.policy_history[-1])

        agent_a.store_experience(state_b, action_a, reward_a, next_state_b)
        agent_b.store_experience(state_a, action_b, reward_b, next_state_a)

        state_a, state_b = next_state_a, next_state_b

        if round % 100 == 0:
            if game.action_space == "discrete":
                action_distribution_a = np.bincount(
                    actions_a[-100:], minlength=agent_a.action_size
                ) / len(actions_a[-100:])
                action_distribution_b = np.bincount(
                    actions_b[-100:], minlength=agent_b.action_size
                ) / len(actions_b[-100:])
            else:
                action_distribution_a, _ = np.histogram(
                    actions_a[-100:], bins=10, range=(0, 10), density=True
                )
                action_distribution_b, _ = np.histogram(
                    actions_b[-100:], bins=10, range=(0, 10), density=True
                )
            action_distributions_a.append(action_distribution_a)
            action_distributions_b.append(action_distribution_b)

        if progress_queue is not None and round % 1000 == 0:
            progress_queue.put(1)

        if round % chunk_size == 0 and round != 0:
            yield outcomes, actions_a, actions_b, action_distributions_a, action_distributions_b, policy_history_a, policy_history_b, agent_a.actor_losses, agent_a.critic_losses, agent_b.actor_losses, agent_b.critic_losses
            outcomes = []
            actions_a = []
            actions_b = []
            action_distributions_a = []
            action_distributions_b = []
            policy_history_a = []
            policy_history_b = []
            agent_a.actor_losses = []
            agent_a.critic_losses = []
            agent_b.actor_losses = []
            agent_b.critic_losses = []
            tf.keras.backend.clear_session()
            gc.collect()

    yield outcomes, actions_a, actions_b, action_distributions_a, action_distributions_b, policy_history_a, policy_history_b, agent_a.actor_losses, agent_a.critic_losses, agent_b.actor_losses, agent_b.critic_losses


# Experiment Runner class with chunking
class ExperimentRunner:
    def __init__(
        self,
        games,
        repetitions=10,
        results_dir="results",
        analysis_dir="analysis",
        rounds_per_experiment=10000,
        base_lr=0.01,
        batch_size=128,
        decay_rate=0.95,
    ):
        self.games = games
        self.repetitions = repetitions
        self.results_dir = results_dir
        self.analysis_dir = analysis_dir
        self.rounds_per_experiment = rounds_per_experiment
        self.base_lr = base_lr
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.create_directory(self.results_dir)
        self.create_directory(self.analysis_dir)
        self.asymmetric_games = [EntryDeterrenceGame, UltimatumGame]

    def create_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def run_single_experiment(
        self, game_class, transparency_a, transparency_b, repetition, progress_queue
    ):
        game_instance = game_class()
        print(
            f"Running experiment for {game_class.__name__}, transparency_a={transparency_a}, transparency_b={transparency_b}, action_space={game_instance.action_space}"
        )

        seed = repetition
        np.random.seed(seed)
        tf.random.set_seed(seed)

        results = {
            "game": game_class.__name__,
            "transparency_a": transparency_a,
            "transparency_b": transparency_b,
            "repetition": repetition,
            "seed": seed,
            "mean_action_a": [],
            "mean_action_b": [],
            "std_dev_a": [],
            "std_dev_b": [],
            "actions_a": [],
            "actions_b": [],
            "convergence_a": [],
            "convergence_b": [],
            "actor_losses_a": [],
            "critic_losses_a": [],
            "actor_losses_b": [],
            "critic_losses_b": [],
            "policy_history_a": [],
            "policy_history_b": [],
        }

        agent_a = BaseAgent(
            state_size=game_instance.state_size,
            action_size=game_instance.action_size,
            action_space=game_instance.action_space,
            base_lr=self.base_lr,
            batch_size=self.batch_size,
            decay_rate=self.decay_rate,
            transparency=transparency_a,
        )
        agent_b = BaseAgent(
            state_size=game_instance.state_size,
            action_size=game_instance.action_size,
            action_space=game_instance.action_space,
            base_lr=self.base_lr,
            batch_size=self.batch_size,
            decay_rate=self.decay_rate,
            transparency=transparency_b,
        )

        chunk_index = 0
        chunk_files = []

        for chunk in simulate_game(
            agent_a,
            agent_b,
            game_instance,
            rounds=self.rounds_per_experiment,
            chunk_size=1000,
            progress_queue=progress_queue,
        ):
            (
                outcomes,
                actions_a,
                actions_b,
                action_distributions_a,
                action_distributions_b,
                policy_history_a,
                policy_history_b,
                actor_losses_a,
                critic_losses_a,
                actor_losses_b,
                critic_losses_b,
            ) = chunk
            results_chunk = {
                "game": game_class.__name__,
                "transparency_a": transparency_a,
                "transparency_b": transparency_b,
                "repetition": repetition,
                "mean_action_a": [float(np.mean(actions_a[-100:]))],
                "mean_action_b": [float(np.mean(actions_b[-100:]))],
                "std_dev_a": [float(np.std(actions_a[-100:]))],
                "std_dev_b": [float(np.std(actions_b[-100:]))],
                "convergence_a": [
                    float(np.std(action_distributions_a[-10:], axis=0).mean())
                ],
                "convergence_b": [
                    float(np.std(action_distributions_b[-10:], axis=0).mean())
                ],
                "actions_a": actions_a,
                "actions_b": actions_b,
                "actor_losses_a": actor_losses_a,
                "critic_losses_a": critic_losses_a,
                "actor_losses_b": actor_losses_b,
                "critic_losses_b": critic_losses_b,
                "policy_history_a": policy_history_a,
                "policy_history_b": policy_history_b,
            }

            chunk_file = self.save_chunk_results(results_chunk, chunk_index)
            chunk_files.append(chunk_file)
            chunk_index += 1

            tf.keras.backend.clear_session()
            gc.collect()

        self.merge_chunks(
            chunk_files,
            results,
            game_class.__name__,
            transparency_a,
            transparency_b,
            repetition,
        )

    def save_chunk_results(self, results, chunk_index):
        filename = f"{results['game']}_tA{results['transparency_a']}_tB{results['transparency_b']}_rep{results['repetition']}_chunk{chunk_index}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json.gz"
        filepath = os.path.join(self.results_dir, filename)
        temp_filepath = filepath + ".tmp"
        try:
            with gzip.open(temp_filepath, "wt", encoding="UTF-8") as f:
                json.dump(convert_to_serializable(results), f)
            os.rename(temp_filepath, filepath)
            return filepath
        except Exception as e:
            print(f"Error saving results: {e}")
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            return None

    def merge_chunks(
        self,
        chunk_files,
        results,
        game_name,
        transparency_a,
        transparency_b,
        repetition,
    ):
        for chunk_file in chunk_files:
            if chunk_file is not None:
                try:
                    with gzip.open(chunk_file, "rt", encoding="UTF-8") as f:
                        chunk_results = json.load(f)

                        results["mean_action_a"].extend(chunk_results["mean_action_a"])
                        results["mean_action_b"].extend(chunk_results["mean_action_b"])
                        results["std_dev_a"].extend(chunk_results["std_dev_a"])
                        results["std_dev_b"].extend(chunk_results["std_dev_b"])
                        results["convergence_a"].extend(chunk_results["convergence_a"])
                        results["convergence_b"].extend(chunk_results["convergence_b"])
                        results["actions_a"].append(chunk_results["actions_a"])
                        results["actions_b"].append(chunk_results["actions_b"])
                        results["actor_losses_a"].append(
                            chunk_results["actor_losses_a"]
                        )
                        results["critic_losses_a"].append(
                            chunk_results["critic_losses_a"]
                        )
                        results["actor_losses_b"].append(
                            chunk_results["actor_losses_b"]
                        )
                        results["critic_losses_b"].append(
                            chunk_results["critic_losses_b"]
                        )
                        results["policy_history_a"].append(
                            chunk_results["policy_history_a"]
                        )
                        results["policy_history_b"].append(
                            chunk_results["policy_history_b"]
                        )

                except Exception as e:
                    print(f"Error reading chunk file {chunk_file}: {e}")

        self.save_results(results)

        for chunk_file in chunk_files:
            if chunk_file is not None and os.path.exists(chunk_file):
                os.remove(chunk_file)

    def run_experiments(self):
        total_experiments = len(self.games) * self.repetitions * 2
        progress_bar = tqdm(total=total_experiments, desc="Running Experiments")
        with multiprocessing.Manager() as manager:
            progress_queue = manager.Queue()
            with ProcessPoolExecutor(
                max_workers=multiprocessing.cpu_count()
            ) as executor:
                futures = []
                for game_class in self.games:
                    transparency_combinations = [(False, False), (True, False)]
                    if game_class in self.asymmetric_games:
                        transparency_combinations.append((False, True))
                    for transparency_a, transparency_b in transparency_combinations:
                        for repetition in range(self.repetitions):
                            future = executor.submit(
                                self.run_single_experiment,
                                game_class,
                                transparency_a,
                                transparency_b,
                                repetition,
                                progress_queue,
                            )
                            futures.append(future)
                while any(future.running() for future in futures):
                    while not progress_queue.empty():
                        progress_queue.get()
                        progress_bar.update()
                    time.sleep(1)
                for future in futures:
                    future.result()
                progress_bar.close()

    def save_results(self, results):
        filename = f"{results['game']}_tA{results['transparency_a']}_tB{results['transparency_b']}_rep{results['repetition']}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json.gz"
        filepath = os.path.join(self.results_dir, filename)
        temp_filepath = filepath + ".tmp"
        try:
            with gzip.open(temp_filepath, "wt", encoding="UTF-8") as f:
                json.dump(convert_to_serializable(results), f)
            os.rename(temp_filepath, filepath)
        except Exception as e:
            print(f"Error saving results: {e}")
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)

    def load_results(self):
        all_results = []
        for file in os.listdir(self.results_dir):
            if file.endswith(".json.gz"):
                try:
                    with gzip.open(
                        os.path.join(self.results_dir, file), "rt", encoding="UTF-8"
                    ) as f:
                        result = json.load(f)
                        all_results.append(result)
                except json.JSONDecodeError as e:
                    print(f"Error reading {file}: {e}")
                except Exception as e:
                    print(f"Unexpected error reading {file}: {e}")
        return all_results


import sys

# Dynamic start method based on OS
if sys.platform == "darwin":  # macOS
    multiprocessing.set_start_method("spawn", force=True)


# Example usage
if __name__ == "__main__":
    games = [GameOfChicken]  # Add your game classes

    experiment_runner = ExperimentRunner(
        games=games,
        repetitions=10,  # Define the number of repetitions for each experiment
        results_dir="results_1",  # Directory to save the results
        analysis_dir="analysis_1",  # Directory to save the analysis later
        rounds_per_experiment=256000,  # Number of rounds for each experiment
        base_lr=0.02,  # Set your desired base learning rate
        batch_size=128,  # Set your desired batch size
        decay_rate=0.95,  # Set your desired decay rate
    )

    # Run the experiments
    experiment_runner.run_experiments()
