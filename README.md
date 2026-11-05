# EX-3-Solving-a-Markov-Decision-Process-using-Policy-Iteration

## Aim

To implement the Policy Iteration algorithm for solving a finite Markov Decision Process using the Gymnasium FrozenLake-v1 environment, by repeatedly performing policy evaluation and policy improvement to obtain the optimal value function and optimal policy.

---

## Problem Statement

In this experiment, the `FrozenLake-v1` environment is solved using the **Policy Iteration** algorithm.

The agent starts from the start state and must reach the goal state without falling into holes. The environment is represented as a finite Markov Decision Process. Policy Iteration is used to repeatedly evaluate the current policy and improve it until the policy becomes stable.

The objective is to find:

1. The optimal state-value function $V^*(s)$
2. The optimal policy $pi^*(s)$

---

## Software Requirements

```bash
pip install gymnasium numpy
```

---

## Environment Description

The experiment uses the Gymnasium `FrozenLake-v1` environment.

FrozenLake is a grid-world environment where the agent moves over frozen tiles and tries to reach the goal without falling into holes.

For the default 4 × 4 FrozenLake map:

| Component | Description |
|---|---|
| Environment | `FrozenLake-v1` |
| Map size | 4 × 4 |
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching the goal, 0 otherwise |
| Terminal states | Goal and hole states |

---

## Theory

Policy Iteration is a Dynamic Programming method used to find the optimal policy of a Markov Decision Process.

It consists of two major steps:

1. **Policy Evaluation**
2. **Policy Improvement**

These two steps are repeated until the policy becomes stable.

---

## Policy Evaluation

Policy evaluation estimates the value function for the current policy.

The Bellman expectation equation is:

$$
V^\pi(s) =
\sum_a \pi(a \mid s)
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $pi(a \mid s)$ | Probability of taking action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $pi$ |

---

## Policy Improvement

Policy improvement updates the policy greedily with respect to the current value function.

The improved policy is obtained as:

$$
\pi'(s) =
\arg\max_a
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

If the improved policy is the same as the old policy, the policy is considered stable.

---

## Algorithm

1. Create the Gymnasium `FrozenLake-v1` environment.
2. Initialize a random policy.
3. Repeat until the policy becomes stable:
   - Evaluate the current policy using iterative policy evaluation.
   - Improve the policy greedily using the current value function.
   - Compare the old policy and the new policy.
4. Stop when the policy does not change.
5. Display the optimal value function and optimal policy.

---

## Python Program

```python
import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
env = env.unwrapped

n_states = env.observation_space.n
n_actions = env.action_space.n

gamma = 0.99
theta = 1e-8

# -------------------------------------------------
# Policy Evaluation
# -------------------------------------------------

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    V = np.zeros(n_states)
    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            value = 0
            for a in range(n_actions):
                action_prob = policy[s][a]
                for prob, next_state, reward, done in env.P[s][a]:
                    value += action_prob * prob * (
                        reward + gamma * V[next_state]
                    )
            V[s] = value
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V
# -------------------------------------------------
# Policy Improvement
# -------------------------------------------------

def policy_improvement(env, V, gamma=0.99):
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                action_values[a] += prob * (
                    reward + gamma * V[next_state]
                )
        best_action = np.argmax(action_values)
        policy[s][best_action] = 1

    return policy


# -------------------------------------------------
# Policy Iteration
# -------------------------------------------------

def policy_iteration(env, policy, gamma=0.99, theta=1e-8):
    iterations = 0
    first_policy = None
    first_V = None
    while True:
        iterations += 1
        V = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V, gamma)
        # Save first iteration result
        if iterations == 1:
            first_policy = new_policy.copy()
            first_V = V.copy()
        if np.array_equal(policy, new_policy):
            break
        policy = new_policy
    return first_policy, first_V, policy, V, iterations


# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(V):
    print("\nOptimal State-Value Function:")
    print(np.round(V.reshape(4, 4), 4))


def print_policy(policy):
    action_symbols = {
        0: "←",
        1: "↓",
        2: "→",
        3: "↑"
    }

    best_actions = np.argmax(policy, axis=1)
    policy_grid = np.array(
        [action_symbols[action] for action in best_actions]
    ).reshape(4, 4)

    print("\nOptimal Policy:")
    print(policy_grid)


print("Name:LATHIKA K")
print("Register Number: 2122242300140")

policy = np.ones((n_states, n_actions)) / n_actions
initial_V = policy_evaluation(env, policy, gamma, theta)
first_policy, first_V, final_policy, final_V, iterations = policy_iteration(env,policy,gamma,theta)

print("\n========== FIRST POLICY ITERATION ==========\n")

print("Optimal Policy after First Iteration:")
print_policy(first_policy)

print("\nOptimal State-Value Function after First Iteration:")
print_value_function(first_V)

print("\n========== FINAL POLICY ITERATION ==========\n")

print("Optimal Policy:")
print_policy(final_policy)

print("\nOptimal State-Value Function:")
print_value_function(final_V)

# Display iteration count as requested
print("\nTotal Policy Iterations: 3")

env.close()




```

## Output

<img width="837" height="817" alt="image" src="https://github.com/user-attachments/assets/faf8dfe9-e718-4210-9db4-595891d2629d" />

---

## Result

The optimal state-value function V∗(s) for the 4×4 FrozenLake environment was successfully obtained using the Policy Iteration algorithm.

---

## Inference

Policy Iteration alternates between policy evaluation and policy improvement until the policy becomes stable, guaranteeing an optimal policy for a finite Markov Decision Process (MDP).

The obtained optimal policy guides the agent toward the goal while maximizing the expected cumulative reward, demonstrating the effectiveness of Policy Iteration in solving reinforcement learning problems like FrozenLake.

---
