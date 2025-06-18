import random 
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import seaborn as sns

class Card:
    """Represents a single playing card."""
    
    def __init__(self, rank):
        self.rank = rank
        # Handles value assignment for face cards
        if rank in ['J', 'Q', 'K']:
            self.value = 10
        elif rank == 'A':
            self.value = 11  # Defaults Ace to 11, but it can be 1 when needed
        else:
            self.value = int(rank) # Numeric value for cards 2-10
    
    # String representation of the card
    def __str__(self):
        return f"{self.rank}"


class Deck:
    """Represents a deck of 52 playing cards."""
    
    # Initializes the deck with an empty list and resets to a full deck
    def __init__(self):
        self.cards = []
        self.reset()
    
    # Resets the deck to a full 52-card deck
    def reset(self):
        """Reset the deck to a full 52-card deck."""
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        # Create a list of cards with 4 suits for each rank
        self.cards = [Card(rank) for rank in ranks for _ in range(4)]
        
    def shuffle(self):
        """Shuffle the deck."""
        random.shuffle(self.cards)
        
    def deal(self):
        """Deal the top card from the deck."""
        if not self.cards:
            raise ValueError("No cards left in the deck")
        return self.cards.pop()


class BlackjackEnv:
    """Blackjack environment implementing the simplified rules from the project brief."""
    
    # Initializes the environment with a deck of new cards, 
    # deals the initial cards, and gets the initial state for the agent
    def __init__(self):
        self.deck = Deck()
        self.reset()
    
    def reset(self):
        """Reset the environment for a new episode."""
        self.deck.reset()
        self.deck.shuffle()
        
        # Deals initial cards
        self.player_cards = [self.deck.deal(), self.deck.deal()]
        self.dealer_cards = [self.deck.deal()]
        
        # Calculates values of the hands of the dealer and agent and handles aces
        self.player_sum = self._calculate_hand_value(self.player_cards)
        self.dealer_sum = self._calculate_hand_value(self.dealer_cards)
        
        # Checks if game is already over (player has 21)
        self.done = (self.player_sum == 21)
        
        # Returns the state representation for the agent
        return self._get_state()
    
    def _calculate_hand_value(self, cards):
        """Calculate the value of a hand, handling aces optimally."""
        non_ace_sum = sum(card.value for card in cards if card.rank != 'A')
        aces = [card for card in cards if card.rank == 'A']
        
        # Starts by counting all aces as 11
        total = non_ace_sum + sum(card.value for card in aces)
        
        # Converts aces from 11 to 1 if needed to stay under 21
        for _ in range(len(aces)):
            if total > 21 and any(card.rank == 'A' and card.value == 11 for card in aces):
                # Finds the first ace still valued at 11
                for ace in aces:
                    if ace.value == 11:
                        ace.value = 1
                        total -= 10
                        break
    
        return total
    
    def _get_state(self):
        """Get the current state representation for the agent."""
        # State is represented as (player_sum, dealer_card_value)
        dealer_card = self.dealer_cards[0]
        dealer_card_value = min(dealer_card.value, 10)  # Treat face cards as 10
        
        return (self.player_sum, dealer_card_value, self._has_usable_ace())
    
    # Checks if player has a usable ace (value 11)
    def _has_usable_ace(self):
        for card in self.player_cards:
            if card.rank == 'A' and card.value == 11:
                return True
        return False
    
    def hit(self):
        """Execute the HIT action and return new state, reward, and whether game is done."""
        
        if self.done:
            return self._get_state(), 0, True
        
        # Deals a new card to the player
        new_card = self.deck.deal()
        self.player_cards.append(new_card)
        
        # Recalculates player's sum
        self.player_sum = self._calculate_hand_value(self.player_cards)
        
        # Checks if player busts
        if self.player_sum > 21:
            self.done = True
            return self._get_state(), -1, True
        
        # Checks if player gets 21
        if self.player_sum == 21:
            self.done = True
            # Need to see what dealer gets
            return self.stand()
        
        return self._get_state(), 0, False
    
    def stand(self):
        """Execute the STAND action and return new state, reward, and whether game is done."""
        if self.done and self.player_sum > 21:  # Player already busted
            return self._get_state(), -1, True
        
        self.done = True # Player stands, game is done
        
        # Dealer plays according to the rules mentioned in assignment
        while self.dealer_sum < 17:
            new_card = self.deck.deal()
            self.dealer_cards.append(new_card)
            self.dealer_sum = self._calculate_hand_value(self.dealer_cards)
        
        # Determines the outcome
        if self.dealer_sum > 21:  # Dealer busts, player wins
            return self._get_state(), 1, True
        elif self.dealer_sum > self.player_sum:  # Dealer wins
            return self._get_state(), -1, True
        elif self.dealer_sum < self.player_sum:  # Player wins
            return self._get_state(), 1, True
        else:  # Draw
            return self._get_state(), 0, True


class BlackjackAgent:
    """Base agent class for learning to play blackjack."""
    
    def __init__(self):
        # Initializes state-action value function Q(s,a)
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Initializes count for state-action pairs N(s,a)
        self.state_action_counts = defaultdict(lambda: defaultdict(int))
        
        # Initializes counter for episodes
        self.episode_count = 0

    # Gets the Q-values for the agent
    @property
    def q_values(self):
        return self.q_table
    
    # Gets the epsilon value for epsilon-greedy policy
    def get_epsilon(self, epsilon_type="1/k"):
        """Get the current epsilon value for epsilon-greedy policy."""
        k = self.episode_count + 1  # Add 1 to avoid division by zero
        
        if epsilon_type == "1/k":
            return 1.0 / k
        elif epsilon_type == "exp-k/1000":
            return np.exp(-k / 1000)
        elif epsilon_type == "exp-k/10000":
            return np.exp(-k / 10000)
        elif epsilon_type == "fixed":
            return 0.1
        else:
            return 1.0 / k  # Default to 1/k
    
    def choose_action(self, state, epsilon_type="1/k", exploring_start=False):
        """Choose an action using epsilon-greedy policy."""
        player_sum, _, _ = state
        
        # Always hits if sum < 12, always stand if sum == 21
        if player_sum < 12:
            return "hit"
        if player_sum == 21:
            return "stand"
        
        # Special case for exploring starts
        if exploring_start and len(self.episode_trajectory) == 0:
            return random.choice(["hit", "stand"])
        
        # Regular epsilon-greedy selection
        epsilon = self.get_epsilon(epsilon_type)
        if random.random() < epsilon:
            return random.choice(["hit", "stand"])
        else:
            # Chooses action with highest Q-value
            hit_value = self.q_values[state]["hit"]
            stand_value = self.q_values[state]["stand"]
            return "hit" if hit_value >= stand_value else "stand"
    
    # Placeholder for updating Q-values in derived classes
    def update_q_value(self, state, action, reward):
        """Will be implemented differently by each algorithm"""
        pass

# Agent simulating Monte Carlo On-Policy Control method 
class MonteCarloAgent(BlackjackAgent):

    # Initializes the agent with epsilon type and exploring starts flag, episode trajectory, and returns
    def __init__(self, epsilon_type="1/k", exploring_starts=False):
        super().__init__()
        self.epsilon_type = epsilon_type  # "1/k", "exp-k/1000", or "exp-k/10000"
        self.exploring_starts = exploring_starts
        self.episode_trajectory = []
        self.returns = defaultdict(lambda: defaultdict(list))
    
    def start_episode(self):
        self.episode_trajectory = []
    
    def observe(self, state, action, reward):
        """Records a state-action-reward observation."""
        self.episode_trajectory.append((state, action, reward))
    
    def update_policy(self):
        self.episode_count += 1
        
        # Implements the algorithm found in the slides

        # Calculates returns
        G = 0
        # Processes the episode in reverse order
        for t in range(len(self.episode_trajectory) - 1, -1, -1):
            state, action, reward = self.episode_trajectory[t]
            
            # Only update Q-values for player sums in [12..20]
            player_sum, _, _ = state
            if player_sum < 12 or player_sum > 20:
                continue
                
            # Calculates the return
            G = reward + G  # No discount (gamma = 1)
            
            # Checks if this state-action pair appears earlier in the episode & only update if this is the first occurrence (first-visit MC)
            if (state, action) not in [(s, a) for s, a, _ in self.episode_trajectory[:t]]:
                # Add the return to the list of returns for this state-action pair
                self.returns[state][action].append(G)
                
                # Update the Q-value to be the average of all returns
                self.q_table[state][action] = sum(self.returns[state][action]) / len(self.returns[state][action])
                
                # Update the count for this state-action pair
                self.state_action_counts[state][action] += 1
                
# Agent simulating SARSA algorithm                
class SARSAAgent(BlackjackAgent):

    # Initializes the agent with epsilon type and exploring starts flag
    def __init__(self, epsilon_type="1/k"):
        super().__init__()
        self.epsilon_type = epsilon_type  # "1/k", "exp-k/1000", or "exp-k/10000"

    # Initializes the last state and action to None
    def start_episode(self):
        self.last_state = None
        self.last_action = None

    def observe(self, state, action, reward):
        """Performs SARSA update after each action taken."""
        if self.last_state is not None and self.last_action is not None:
            # Gets current Q value
            current_q = self.q_values[self.last_state][self.last_action]

            # Calculate the step size (α = 1 / (N(s,a) + 1)
            count = self.state_action_counts[self.last_state][self.last_action]
            alpha = 1.0 / (count + 1)
        
                        # Estimate next Q-value
            next_q = self.q_table[state][action]

            # SARSA Update Rule:
            updated_q = current_q + alpha * (reward + next_q - current_q)
            self.q_values[self.last_state][self.last_action] = updated_q

            # Increment count
            self.state_action_counts[self.last_state][self.last_action] += 1

        # Update last state-action
        self.last_state = state
        self.last_action = action
    
    def update_policy(self):
        """At end of episode, finalize last update."""
        # For terminal state, expected future reward is 0
        if self.last_state is not None and self.last_action is not None:
            current_q = self.q_values[self.last_state][self.last_action]
            count = self.state_action_counts[self.last_state][self.last_action]
            alpha = 1.0 / (count + 1)

            # Final update with reward only
            updated_q = current_q + alpha * (-current_q)
            self.q_values[self.last_state][self.last_action] = updated_q
            self.state_action_counts[self.last_state][self.last_action] += 1
        self.episode_count += 1

# SARSAAgent algorithm but maximizes Q-value for next action
class QLearningAgent(BlackjackAgent):

    def __init__(self, epsilon_type="1/k"):
        super().__init__()
        self.epsilon_type = epsilon_type  # "1/k", "exp-k/1000", or "exp-k/10000"

    def start_episode(self):
        self.last_state = None
        self.last_action = None

    def observe(self, state, action, reward):
        """Performs Q-Learning update after each action taken."""
        if self.last_state is not None and self.last_action is not None:
            current_q = self.q_values[self.last_state][self.last_action]

            count = self.state_action_counts[self.last_state][self.last_action]
            alpha = 1.0 / (count + 1)
        
            # Q-Learning uses max_a' Q(s', a')
            next_hit = self.q_values[state]["hit"]
            next_stand = self.q_values[state]["stand"]
            max_next_q = max(next_hit, next_stand)

            updated_q = current_q + alpha * (reward + max_next_q - current_q)
            self.q_values[self.last_state][self.last_action] = updated_q

            self.state_action_counts[self.last_state][self.last_action] += 1

        self.last_state = state
        self.last_action = action
    
    def update_policy(self):
        """At end of episode, finalize last update."""
        # For terminal state, expected future reward is 0
        if self.last_state is not None and self.last_action is not None:
            current_q = self.q_values[self.last_state][self.last_action]
            count = self.state_action_counts[self.last_state][self.last_action]
            alpha = 1.0 / (count + 1)

            # Final update with reward only
            updated_q = current_q + alpha * (-current_q)
            self.q_values[self.last_state][self.last_action] = updated_q
            self.state_action_counts[self.last_state][self.last_action] += 1
        self.episode_count += 1

# QLearning agent but uses two Q-value tables for Double Q-Learning
class DoubleQLearningAgent(BlackjackAgent):
    def __init__(self, epsilon_type="0.1"):
        super().__init__()
        self.epsilon_type = epsilon_type
        # Initializes two Q-value tables for Double Q-Learning
        self.q1_values = defaultdict(lambda: defaultdict(float))
        self.q2_values = defaultdict(lambda: defaultdict(float))

    def start_episode(self):
        self.last_state = None
        self.last_action = None

    def observe(self, state, action, reward):
        """Performs Double Q-Learning update after each action."""
        if self.last_state is not None and self.last_action is not None:
            # Decides with 50% chance which Q to update
            if random.random() < 0.5:
                # Update q1 using q2 for evaluation
                current_q = self.q1_values[self.last_state][self.last_action]
                count = self.state_action_counts[self.last_state][self.last_action]
                alpha = 1.0 / (count + 1)

                # Select action using q1
                next_action = self._argmax_q(self.q1_values, state)
                target = reward + self.q2_values[state][next_action]

                self.q1_values[self.last_state][self.last_action] += alpha * (target - current_q)
            else:
                # Update q2 using q1 for evaluation
                current_q = self.q2_values[self.last_state][self.last_action]
                count = self.state_action_counts[self.last_state][self.last_action]
                alpha = 1.0 / (count + 1)

                next_action = self._argmax_q(self.q2_values, state)
                target = reward + self.q1_values[state][next_action]

                self.q2_values[self.last_state][self.last_action] += alpha * (target - current_q)

            # Increments count
            self.state_action_counts[self.last_state][self.last_action] += 1

        self.last_state = state
        self.last_action = action
 
    def _argmax_q(self, q_values, state):
        """Returns action with highest Q-value from given table."""
        return "hit" if q_values[state]["hit"] >= q_values[state]["stand"] else "stand"

    def update_policy(self):
        """Finalize update at terminal state (no next state)."""
        if self.last_state is not None and self.last_action is not None:
            count = self.state_action_counts[self.last_state][self.last_action]
            alpha = 1.0 / (count + 1)

            # Finalize both q1 and q2 updates at terminal (no future reward)
            self.q1_values[self.last_state][self.last_action] += alpha * (-self.q1_values[self.last_state][self.last_action])
            self.q2_values[self.last_state][self.last_action] += alpha * (-self.q2_values[self.last_state][self.last_action])

            self.state_action_counts[self.last_state][self.last_action] += 1

        self.episode_count += 1

    # @property makes q_values be accessed like an attribute instead of a method
    @property
    def q_values(self):
        """Return the averaged Q-values for decision making."""
        averaged_q = defaultdict(lambda: defaultdict(float))
        for state in set(list(self.q1_values.keys()) + list(self.q2_values.keys())):
            for action in ["hit", "stand"]:
                q1 = self.q1_values[state][action]
                q2 = self.q2_values[state][action]
                averaged_q[state][action] = 0.5 * (q1 + q2)
        return averaged_q

# BlackjackSimulator class to run the environment and agent
class BlackjackSimulator:
    def __init__(self, agent, env=None):
        self.agent = agent
        self.env = env if env else BlackjackEnv()
        self.results = {
            'wins': [],
            'losses': [],
            'draws': []
        }

        # For keeping track of the last 1000 episodes
        self.current_wins = 0
        self.current_losses = 0
        self.current_draws = 0
    
    def run_episode(self):
        """Runs a single episode of Blackjack."""
        
        # Resets the environment and agent
        state = self.env.reset()
        self.agent.start_episode()
        done = False
        
        # Choose actions until the episode is done
        while not done:
            action = self.agent.choose_action(
                state, 
                epsilon_type=self.agent.epsilon_type if hasattr(self.agent, 'epsilon_type') else "1/k",
                exploring_start=self.agent.exploring_starts if hasattr(self.agent, 'exploring_starts') else False
            )
            
            # Carry out the action
            if action == "hit":
                next_state, reward, done = self.env.hit()
            else: 
                next_state, reward, done = self.env.stand()
            
            # Record observation and update state
            self.agent.observe(state, action, reward)
            state = next_state
        
        # Update agent's policy based on the completed episode
        self.agent.update_policy()
        
        if hasattr(self.agent, 'episode_trajectory'):
            final_reward = self.agent.episode_trajectory[-1][2]
        else:
            final_reward = reward   # Use last reward for SARSA
        if final_reward == 1:
            self.current_wins += 1
            return "win"
        elif final_reward == -1:
            self.current_losses += 1
            return "loss"
        else:
            self.current_draws += 1
            return "draw"
    
    def run_simulations(self, num_episodes):
        """Run multiple episodes and collect statistics."""
        for i in range(1, num_episodes + 1):
            self.run_episode()
            
            # Record statistics every 1000 episodes
            if i % 1000 == 0:
                self.results['wins'].append(self.current_wins)
                self.results['losses'].append(self.current_losses)
                self.results['draws'].append(self.current_draws)
                self.current_wins = 0
                self.current_losses = 0
                self.current_draws = 0
                
                # Print progress
                if i % 25000 == 0:
                    print(f"Completed {i} episodes...")
        
        return self.results
    
    def get_best_policy(self):
        """Generate the best policy based on learned Q-values."""

        # Initialize policy table one for each case (usable ace and no usable ace)
        policy_usable_ace = np.full((9, 10), "S", dtype=str)
        policy_no_ace = np.full((9, 10), "S", dtype=str)
        
        # Fill in the policy table based on learned Q-values
        for player_sum in range(12, 21):
            for dealer_card in range(2, 12):
                for has_usable_ace in [True, False]:
                    state = (player_sum, dealer_card, has_usable_ace)
                    hit_value = self.agent.q_values[state]["hit"]
                    stand_value = self.agent.q_values[state]["stand"]
                    best_action = "H" if hit_value > stand_value else "S"
                    
                    if has_usable_ace:
                        policy_usable_ace[player_sum - 12][dealer_card - 2] = best_action
                    else: 
                        policy_no_ace[player_sum - 12][dealer_card - 2] = best_action
        
        return policy_usable_ace, policy_no_ace
    
    def get_state_action_counts(self):
        """Get counts for each state-action pair."""
        counts = []
        for state in self.agent.state_action_counts:
            for action in self.agent.state_action_counts[state]:
                count = self.agent.state_action_counts[state][action]
                counts.append((state, action, count))
        
        # Sort by count (descending)
        counts.sort(key=lambda x: x[2], reverse=True)
        return counts
    
    def get_unique_state_action_pairs(self):
        """Get the number of unique state-action pairs explored."""
        count = 0
        for state in self.agent.state_action_counts:
            count += len(self.agent.state_action_counts[state])
        return count
    
    def calculate_dealer_advantage(self):
        """Calculate the dealer advantage."""
        wins = sum(self.results['wins'][-10:])
        losses = sum(self.results['losses'][-10:])
        
        if wins + losses == 0:
            return 0
        
        return (losses - wins) / (losses + wins)


class Evaluator:
    """Class for evaluating and visualizing RL algorithm results."""
    
    def __init__(self):
        self.all_results = {}
    
    def add_result(self, algorithm_name, config_name, simulator):
        """Add results from a simulator run to the collection."""
        # Create a unique key for this algorithm + configuration
        key = f"{algorithm_name} - {config_name}"
        
        # Get policy table
        policy_usable_ace, policy_no_ace = simulator.get_best_policy()
        
        # Save results
        self.all_results[key] = {
            "simulator": simulator,
            "stats": simulator.results,
            "policy_usable_ace": policy_usable_ace,
            "policy_no_ace": policy_no_ace,
            "state_action_counts": simulator.get_state_action_counts(),
            "unique_pairs": simulator.get_unique_state_action_pairs(),
            "dealer_advantage": simulator.calculate_dealer_advantage(),
            "algorithm": algorithm_name,
            "config": config_name
        }
    
    def plot_wins_losses_draws(self):
        for key, result in self.all_results.items():
            plt.figure(figsize=(10, 6))
            
            # Get statistics
            stats = result["stats"]
            episodes = np.arange(1000, 1000 * (len(stats["wins"]) + 1), 1000)
            
            # Plot wins, losses & draws
            plt.plot(episodes, stats["wins"], label="Wins", color="green")
            plt.plot(episodes, stats["losses"], label="Losses", color="red")
            plt.plot(episodes, stats["draws"], label="Draws", color="blue")
            
            plt.title(f"Performance over Episodes: {key}")
            plt.xlabel("Episodes")
            plt.ylabel("Count (per 1000 episodes)")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.savefig(f"images/wins_losses_draws_{key.replace(' ', '_').replace('/', '_')}.png")
            plt.show()
            
    
    def plot_state_action_counts(self):
        for key, result in self.all_results.items():
            # Gets top 30 state-action pairs by count
            counts = result["state_action_counts"][:30]
            
            if not counts:  # Skip if no counts
                continue
                
            # Prepares data for plotting
            labels = [f"{state[0]},{state[1]},{'A' if state[2] else 'NA'}-{action}" for state, action, _ in counts]
            values = [count for _, _, count in counts]
            
            plt.figure(figsize=(12, 8))
            plt.bar(range(len(values)), values, color="skyblue")
            plt.xticks(range(len(labels)), labels, rotation=90)
            plt.title(f"Top State-Action Pair Counts: {key}")
            plt.xlabel("State-Action Pair")
            plt.ylabel("Count")
            plt.tight_layout()

            plt.savefig(f"images/state_action_counts_{key.replace(' ', '_').replace('/', '_')}.png")
            plt.show()
    
    def plot_unique_state_action_pairs(self):
        """Plot the total number of unique state-action pairs for each algorithm configuration."""
          
        # Group by algorithm
        algorithms = {}
        for key, result in self.all_results.items():
            algorithm = result["algorithm"]
            if algorithm not in algorithms:
                algorithms[algorithm] = []
            algorithms[algorithm].append((result["config"], result["unique_pairs"]))
        
        # Plot per algorithm
        for algorithm, configs in algorithms.items():
            plt.figure(figsize=(10, 6))
            
            configs.sort() 
            labels = [config for config, _ in configs]
            values = [count for _, count in configs]
            
            plt.bar(range(len(values)), values, color="lightgreen")
            plt.xticks(range(len(labels)), labels, rotation=45)
            plt.title(f"Unique State-Action Pairs Explored: {algorithm}")
            plt.xlabel("Configuration")
            plt.ylabel("Count")
            plt.tight_layout()

            plt.savefig(f"images/unique_state_action_pairs_{algorithm.replace(' ', '_').replace('/', '_')}.png")
            plt.show()

    def plot_dealer_advantage(self):
        """Plot the dealer advantage for each algorithm configuration."""
        
        # Get unique algorithms for filename
        unique_algorithms = "_".join(sorted(set(result["algorithm"].replace(" ", "_") for result in self.all_results.values())))
        
        algorithms = set()
        configs = set()
        data = []
        
        for key, result in self.all_results.items():
            algorithm = result["algorithm"]
            config = result["config"]
            advantage = result["dealer_advantage"]
            
            algorithms.add(algorithm)
            configs.add(config)
            data.append((algorithm, config, advantage))
        
        df = pd.DataFrame(data, columns=["Algorithm", "Configuration", "Dealer Advantage"])
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x="Algorithm", y="Dealer Advantage", hue="Configuration", data=df)
        plt.title("Dealer Advantage by Algorithm and Configuration")
        plt.xlabel("Algorithm")
        plt.ylabel("Dealer Advantage")
        plt.legend(title="Configuration")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(f"images/dealer_advantage_{unique_algorithms}.png")
        plt.show()
    
    def visualize_policy(self):
        """Visualize the policy tables for each algorithm configuration."""
        # Iterate through all results and visualize policies
        for key, result in self.all_results.items():
            # Get the policy tables (usable ace and no ace)
            for ace_flag, policy in [("Usable Ace", result["policy_usable_ace"]),
                                 ("No Usable Ace", result["policy_no_ace"])]:
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 8))
            
                # Set up the data
                player_sums = list(range(20, 11, -1))
                dealer_cards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']
            
                # Visualize policy
                table = ax.table(
                    cellText=policy[::-1],
                    rowLabels=player_sums,
                    colLabels=dealer_cards,
                    cellLoc='center',
                    loc='center'
                )
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.scale(1.2, 2)
                ax.set_title(f"Optimal Policy ({ace_flag}): {key}")
                ax.axis('off')
            
                plt.tight_layout()
                plt.savefig(f"images/policy_{ace_flag.replace(' ', '_').lower()}_{key.replace(' ', '_').replace('/', '_')}.png")
                plt.show()
    
    def print_summary(self):
        """Print a summary of the results."""
        print("\n" + "="*50)
        print("SUMMARY OF RESULTS")
        print("="*50)
        
        for key, result in self.all_results.items():
            print(f"\n{key}:")
            print(f"  Unique state-action pairs explored: {result['unique_pairs']}")
            
            # Calculate win/loss/draw percentages for the last 10,000 episodes
            wins = sum(result['stats']['wins'][-10:])
            losses = sum(result['stats']['losses'][-10:])
            draws = sum(result['stats']['draws'][-10:])
            total = wins + losses + draws
            
            if total > 0:
                win_pct = wins / total * 100
                loss_pct = losses / total * 100
                draw_pct = draws / total * 100
                
                print(f"  Last 10,000 episodes:")
                print(f"    Wins: {wins} ({win_pct:.1f}%)")
                print(f"    Losses: {losses} ({loss_pct:.1f}%)")
                print(f"    Draws: {draws} ({draw_pct:.1f}%)")
                print(f"    Dealer advantage: {result['dealer_advantage']:.4f}")
            
            print("-"*30)


def run_monte_carlo_simulations():
    """Run MC simulations with different configurations and evaluate results."""
    evaluator = Evaluator()
    
    # Define the configs for MC
    configurations = [
        {"name": "Exploring Starts, epsilon=1/k", "exploring_starts": True, "epsilon_type": "1/k"},
        {"name": "No Exploring Starts, epsilon=1/k", "exploring_starts": False, "epsilon_type": "1/k"},
        {"name": "No Exploring Starts, epsilon=exp(-k/1000)", "exploring_starts": False, "epsilon_type": "exp-k/1000"},
        {"name": "No Exploring Starts, epsilon=exp(-k/10000)", "exploring_starts": False, "epsilon_type": "exp-k/10000"}
    ]
    
    # Run simulations for each configuration
    for config in configurations:
        print(f"\nRunning Monte Carlo with {config['name']}...")
        
        # Create agent and simulator
        agent = MonteCarloAgent(
            epsilon_type=config["epsilon_type"],
            exploring_starts=config["exploring_starts"]
        )
        simulator = BlackjackSimulator(agent)
        simulator.run_simulations(100000)
      
        # Add results to evaluator
        evaluator.add_result("Monte Carlo", config["name"], simulator)
    
    # Generate plots and analysis
    print("\nGenerating evaluation plots...")
    evaluator.plot_wins_losses_draws()
    evaluator.plot_state_action_counts()
    evaluator.plot_unique_state_action_pairs()
    evaluator.plot_dealer_advantage()
    evaluator.visualize_policy()
    
    # Print summary of results
    evaluator.print_summary()
    
    return evaluator

def run_sarsa_simulations():
    """Run SARSA simulations with different configurations and evaluate results."""
    evaluator = Evaluator()
    
    # Define the configs for SARSA
    configurations = [
        {"name": "epsilon=0.1", "epsilon_type": "fixed"},
        {"name": "epsilon=1/k", "epsilon_type": "1/k"},
        {"name": "epsilon=exp(-k/1000)", "epsilon_type": "exp-k/1000"},
        {"name": "epsilon=exp(-k/10000)", "epsilon_type": "exp-k/10000"}
    ]
    
    for config in configurations:
        print(f"\nRunning SARSA with {config['name']}...")
        
        agent = SARSAAgent(epsilon_type=config["epsilon_type"])
        simulator = BlackjackSimulator(agent)
        simulator.run_simulations(100000)
        
        evaluator.add_result("SARSA", config["name"], simulator)
    
    # Generate evaluation plots
    print("\nGenerating SARSA evaluation plots...")
    evaluator.plot_wins_losses_draws()
    evaluator.plot_state_action_counts()
    evaluator.plot_unique_state_action_pairs()
    evaluator.plot_dealer_advantage()
    evaluator.visualize_policy()
    
    evaluator.print_summary()
    
    return evaluator

def run_q_learning_simulations():
    """Run Q-Learning simulations with different configurations and evaluate results."""
    evaluator = Evaluator()
    
    configurations = [
        {"name": "epsilon=0.1", "epsilon_type": "fixed"},
        {"name": "epsilon=1/k", "epsilon_type": "1/k"},
        {"name": "epsilon=exp(-k/1000)", "epsilon_type": "exp-k/1000"},
        {"name": "epsilon=exp(-k/10000)", "epsilon_type": "exp-k/10000"}
    ]
    
    for config in configurations:
        print(f"\nRunning Q-Learning with {config['name']}...")
        
        agent = QLearningAgent(epsilon_type=config["epsilon_type"])
        simulator = BlackjackSimulator(agent)
        simulator.run_simulations(100000)
        
        evaluator.add_result("Q-Learning", config["name"], simulator)
    
    # Generate evaluation plots
    print("\nGenerating Q-Learning evaluation plots...")
    evaluator.plot_wins_losses_draws()
    evaluator.plot_state_action_counts()
    evaluator.plot_unique_state_action_pairs()
    evaluator.plot_dealer_advantage()
    evaluator.visualize_policy()
    
    evaluator.print_summary()
    
    return evaluator

def run_double_q_learning_simulations():
    evaluator = Evaluator()
    
    configurations = [
        {"name": "epsilon=0.1", "epsilon_type": "fixed"},
        {"name": "epsilon=1/k", "epsilon_type": "1/k"},
        {"name": "epsilon=exp(-k/1000)", "epsilon_type": "exp-k/1000"},
        {"name": "epsilon=exp(-k/10000)", "epsilon_type": "exp-k/10000"}
    ]
    
    for config in configurations:
        print(f"\nRunning Double Q-Learning with {config['name']}...")
        
        agent = DoubleQLearningAgent(epsilon_type=config["epsilon_type"])
        simulator = BlackjackSimulator(agent)
        simulator.run_simulations(100000)
        
        evaluator.add_result("Double Q-Learning", config["name"], simulator)
    
    evaluator.plot_wins_losses_draws()
    evaluator.plot_state_action_counts()
    evaluator.plot_unique_state_action_pairs()
    evaluator.plot_dealer_advantage()
    evaluator.visualize_policy()
    evaluator.print_summary()
    
    return evaluator

if __name__ == "__main__":

    run_monte_carlo_simulations()
    print("\nSimulations and evaluation complete!")

    run_sarsa_simulations()
    print("\nSARSA Simulations and evaluation complete!")

    run_q_learning_simulations()
    print("\nQ-Learning Simulations and evaluation complete!")

    run_double_q_learning_simulations()
    print("\nDouble Q-Learning Simulations and evaluation complete!")