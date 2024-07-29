# Master Thesis Transparent Games Actor Critic

Repository for master thesis work on TIØ4912 Spring 2024

## Overview

This repository contains the code for simulating and analyzing various game theory scenarios using actor-critic methods with transparency mechanisms. The goal is to understand how transparency affects agent behavior and outcomes in different strategic games.

## Repository Structure

The repository is organized as follows:

```
actor_critic_transparent_games/
│
├── experiment.py         # Main experiment script defining game classes and running simulations
├── process_data.ipynb    # Jupyter notebook for aggregating and analyzing results
├── requirements.txt      # List of required Python packages
```

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Experiment Details](#experiment-details)
4. [Data Aggregation](#data-aggregation)
5. [Results Visualization](#results-visualization)
6. [Contributing](#contributing)
7. [License](#license)

## Installation

To set up the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/actor_critic_transparent_games.git
   cd actor_critic_transparent_games
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows, use `venv\Scriptsctivate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Install `pipreqs` to manage dependencies:

   ```bash
   pip install pipreqs
   ```

5. Generate `requirements.txt` from the current environment:
   ```bash
   pipreqs .
   ```

## Usage

To run the experiments, use the `experiment.py` script. Below is an example setup:

```python
from experiment import GameOfChicken, ExperimentRunner

games = [GameOfChicken] # Add your game classes here

experiment_runner = ExperimentRunner(
    games=games,
    repetitions=10,
    results_dir="results",
    analysis_dir="analysis",
    rounds_per_experiment=256000,
    base_lr=0.02,
    batch_size=128,
    decay_rate=0.95,
)

# Run the experiments
experiment_runner.run_experiments()
```

Or run them by running the experiment.oy files after modifying the main function at the bottom there.

## Experiment Details

The main experiment script `experiment.py` defines several game classes and an `ExperimentRunner` class to automate the experiments.

### Game Classes

- `GameOfChicken`
- `PrisonersDilemma`
- `StagHunt`
- `EntryDeterrenceGame`
- `UltimatumGame`
- `CoordinationGame`
- `BertrandCompetition`
- `PublicGoodsGame`

And also new potential games to run not done in thesis work:

- `BargainingGame`
- `MatchingPennies`
- `RockPaperScissors`
- `BattleOfTheSexes`

### Agent Class

The `BaseAgent` class implements the actor-critic algorithm with options for transparency.

## Data Aggregation

To aggregate and analyze the results, use the `process_data.ipynb` notebook. It includes functions to load, flatten, and process the experiment data. The results are saved in `.parquet` files and can be loaded and analyzed using pandas and other tools.
