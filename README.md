# N-Puzzle Problem Solver with Simulated Annealing

I created this project for my CAP6635 - Advanced AI class, taught by Dr. Ayan Dutta. The program in this repository implements the Simulated Annealing (SA) algorithm 
to solve N-puzzle problems for N in the range 8 to 99, both inclusive, where N + 1 is a perfect square. It includes both linear and geometric cooling scheduler implementations
to find more efficient and effective ways to map from time to temperature. It uses a combined heuristic of the sum of the manhattan distance of each tile from its 
desired tile location + the number of misplaced tiles. Overall, I found the Geometric Scheduler to be more effective and efficient, at least on size 8 and 15 problems.

## Key Features and Summary 
- **Simulated Annealing Algorithm**: Utilize SA to solve the N-puzzle problem for perfect square N + 1 where N is in the range 8 to 99.
- **Cooling Schedules**: Includes both linear and geometric scheduler class implementations which can be used interchangeably by the SA function.
- **Customizable Parameters**: SA variables are highly customizable, meaning users can select their own starting temperature, cooling rate (linear), alpha value (geometric), and min temperature (geometric).
- **Interactive Prompt**: Interactive command line interfact allows for easy experimentation of different values.
- **Puzzle Input**: The initial puzzle states are read from input text files.

## Getting Started

### Prerequisites

Ensure Python 3.x is installed on your system. You can download Python [here](https://www.python.org/downloads/).

### Installation

1. Clone the repository:

```bash
bash git clone https://github.com/yourusername/yourrepositoryname.git
```

2. Navigate to the project directory:
```bash
cd yourrepositoryname
```

### Usage

Run the main script using Python:
```bash
bash python main.py
```

Follow the interactive prompts to configure the SA algorithm and solve an N-puzzle problem. Here are two examples for both schedulers.

Linear Scheduler:

Geometric Scheduler:

## Future Work

Implement more advanced heuristics to better estimate state value, and improve algorithm effectiveness and efficiency. 
Potentially implement a GUI for better visualization of the problem and solution.

# Acknowledgements
Dr. Ayan Dutta for introducing us to the N-Puzzle problem, as well as some of the various algorithms and heuristics used to solve it.
https://aima.cs.berkeley.edu/ for their Simulated Annealing psuedocode.
