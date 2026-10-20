# 🚀 Space Explorer AI
### A Comparative Study of Q-Learning and SARSA for Intelligent Space Navigation

## 📌 About the Project

Space Explorer AI is a Python-based reinforcement learning project that simulates an intelligent agent navigating through a space environment. The agent must collect energy crystals, avoid dangerous asteroids, and reach the destination planet while maximizing its total reward.

This project implements and compares two reinforcement learning algorithms: **Q-Learning** and **SARSA**. Their performance is evaluated using rewards, success rates, steps taken, and collision rates.

## 🎯 Objectives

- Understand reinforcement learning through a practical simulation.
- Implement Q-Learning and SARSA algorithms.
- Train an intelligent agent to navigate a grid-based space environment.
- Compare the learning performance of both algorithms.
- Visualize training results and performance metrics.

## ✨ Features

- Interactive space navigation using Pygame.
- Q-Learning and SARSA implementation.
- Training and evaluation modes.
- Energy crystal collection and reward calculation.
- Asteroid avoidance and destination navigation.
- Performance comparison charts and dashboard.
- Evaluation using multiple performance metrics.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Game Interface:** Pygame
- **Numerical Computation:** NumPy
- **Data Visualization:** Matplotlib
- **Concept:** Reinforcement Learning

## 🧠 Algorithms

### 1. Q-Learning

Q-Learning is an off-policy reinforcement learning algorithm that learns action values using the maximum estimated future reward.

### 2. SARSA

SARSA is an on-policy reinforcement learning algorithm that updates action values using the next action selected by the current policy.

Both algorithms learn through interaction with the environment and aim to improve navigation decisions.

## 🎮 Environment and Rewards

The agent navigates a grid-based space environment with the following elements:

- **Energy Crystals:** Positive rewards for collection.
- **Destination Planet:** Reward for successfully completing the mission.
- **Asteroids:** Negative rewards and mission termination upon collision.
- **Movement:** A small penalty encourages efficient navigation.
- **Step Limit:** Prevents the agent from navigating indefinitely.

## 📊 Performance Evaluation

The algorithms can be compared using the following metrics:

- **Cumulative Reward:** Total reward obtained during an episode.
- **Success Rate:** Percentage of episodes that reach the destination.
- **Steps to Success:** Number of steps required to complete a successful mission.
- **Collision Rate:** Percentage of episodes that end in asteroid collisions.
- **Crystals Collected:** Average number of energy crystals collected.

## 📂 Project Structure

```text
Space-Explorer-AI/
│
├── main.py
├── config.py
├── comparison_dashboard.py
├── web_app.py
│
├── environment/
│   └── space_environment.py
│
├── agents/
│   ├── q_learning.py
│   └── sarsa.py
│
├── training/
│   └── trainer.py
│
├── evaluation/
│   ├── evaluate_agents.py
│   └── plots.py
│
├── utils/
│   └── storage.py
│
├── results/
├── models/
├── requirements.txt
└── README.md
```

*Note: This is an example structure. Keep only the folders and files that actually exist in your project.*

## ⚙️ Installation and Setup

### Step 1: Install Python

Install Python from [python.org](https://www.python.org/downloads/).

### Step 2: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Space-Explorer-AI.git
cd Space-Explorer-AI
```

Replace `YOUR-USERNAME` with your GitHub username.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Project

Run the main program using:

```bash
python main.py
```

Use the commands below only if the corresponding scripts and options are implemented in your project:

```bash
python comparison_dashboard.py
python web_app.py
```

## 📈 Expected Outcome

The project demonstrates how Q-Learning and SARSA learn navigation strategies through repeated interactions with an environment. Their performance metrics and visualizations help analyze differences in learning behavior and mission completion.

Actual results depend on the training settings, random seeds, and environment configuration.

## 🔮 Future Enhancements

- Introduce larger and more complex space environments.
- Add dynamic obstacles and new reward challenges.
- Improve visualization of the agent's learning process.
- Experiment with additional reinforcement learning algorithms.
- Develop more detailed performance reports.

## 👩‍💻 Author

**Pavithra P**  
Information Technology

## 📄 License

This project is intended for educational and research purposes.
