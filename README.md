# Blackjack Reinforcement Learning Simulator 🃏🧠

This project simulates the game of Blackjack using various reinforcement learning algorithms to learn optimal policies. It includes implementations of **Monte Carlo**, **SARSA**, **Q-Learning**, and **Double Q-Learning** agents. The agents learn through repeated simulated games and are evaluated based on their performance.

## Features

- Custom `BlackjackEnv` environment with simplified rules.
- Four learning agents:
  - Monte Carlo (on-policy, with/without exploring starts)
  - SARSA
  - Q-Learning
  - Double Q-Learning
- Dynamic epsilon policies (`1/k`, `exp(-k/1000)`, etc.)
- Visualizations:
  - Win/loss/draw trends
  - Optimal policy heatmaps
  - State-action visit counts
  - Dealer advantage comparison
- Evaluation and summary reports across configurations

## Requirements

- Python 3.10.11

### Instructions

```bash
git clone https://github.com/LuigiCamilleri05/blackjack-ai.git
cd blackjack-ai
```

Install requirements
```bash
pip install -r requirements.txt
```

Run the Code
```bash
python blackjack.py
```

## 🔄 What This Script Does

- Trains all four agents
- Evaluates performance
- Generates plots and policy visualizations
- Saves charts to the `images/` folder

> ⏱️ Each agent performs **100,000 simulated games** per configuration.

---

## 📁 Project Structure

- **`BlackjackEnv`**: Game logic and reward mechanics
- **`BlackjackAgent`**: Base class for all agents
- **`MonteCarloAgent`**, **`SARSAAgent`**, **`QLearningAgent`**, **`DoubleQLearningAgent`**: Implement learning strategies
- **`BlackjackSimulator`**: Manages training episodes and game flow
- **`Evaluator`**: Handles visualization and performance reporting

---

## 📊 Output Examples

Policy tables and performance graphs are saved as PNG files in the `/images` folder. These include:

![image](https://github.com/user-attachments/assets/9f339992-6016-4f4d-8427-42162f035d40)
![image](https://github.com/user-attachments/assets/1d053aa3-2fb5-4547-894f-8ad0466161a8)
![image](https://github.com/user-attachments/assets/b80ec389-cf08-476d-889b-bd3115227963)



---
