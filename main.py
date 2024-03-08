
import math
import os
import itertools
import random
import time

# For reproducability during testing
# random.seed(42)

# Parent class for more specific scheduler methods.
class Scheduler: 
    def __init__(self, starting_temp=10000):
        self.starting_temp = starting_temp 
        self.current_temperature = starting_temp

    # Method to be overwriten by specific schedulers
    def get_new_temperature(self, t):
        raise NotImplementedError("Other schedulers should provide their own implementation of this function.")

class LinearScheduler(Scheduler):
    """
        Linear scheduler subclass for mapping between time and temperature for the SA algorithm.
        It does this by calculting current temp through subtracting from the starting temp the cooling rate times the iteration number (time).
    """
    def __init__(self, starting_temp=10000, cooling_rate=1):
        super().__init__(starting_temp)
        self.cooling_rate = cooling_rate


    def get_new_temperature(self, t): # t stands for time
        self.current_temperature = self.starting_temp - (self.cooling_rate * t)

        if self.current_temperature < 0:
            return 0
        else:
            return self.current_temperature
        
class GeometricScheduler(Scheduler):
    """
        Maps between time t and temperature for the SA algorithm.
        It does this by multiplying current temperature by a constant factor alpha 
        every iteration of the SA algorithm.
    """
    def __init__(self, starting_temp=10000, alpha=0.95, min_temperature=1):
        super().__init__(starting_temp)
        self.alpha = alpha
        self.min_temperature = min_temperature

    # When current temperature times alpha goes below the minimum allowed temperature, this function will return 0.
    # This will stop the SA algorithm.
    def get_new_temperature(self, t): # t stands for time
        self.current_temperature *= self.alpha

        if self.current_temperature < self.min_temperature:
            return 0
        else:
            return self.current_temperature
        

def read_puzzle_from_input(file_path, square_side_length, n_value):

    # Store the rows as a 2D Matrix to form a square grid
    puzzle_rows = []

    # Prevent duplicate numbers from being added to the puzzle
    existing_numbers = set()

    with open(file_path, 'r') as rows:
        
        for i, row in enumerate(rows):     

            # Remove leading and trailing white space and convert string to list
            row_as_list = row.strip().split()

            if len(row_as_list) != square_side_length:
                raise ValueError(f"Row number {i + 1} in {file_path} has a different number of elements than the needed side length {square_side_length}")

            # Convert elements to int for value checking
            row_as_ints = [int(num) for num in row_as_list]
            
            for j, num in enumerate(row_as_ints):
                if num < 0 or num > n_value:
                    raise ValueError(f"The number at position {j + 1} in row {i + 1} is not in the range between 0 and n value")
                elif num in existing_numbers:
                    raise ValueError(f"The number at position {j + 1} in row {i + 1} has already been added to the puzzle, it can't be added twice.")
                else:
                    existing_numbers.add(num)
                
            puzzle_rows.append(row_as_ints)
    
    if len(puzzle_rows) != square_side_length: 
        print(f"The number of rows: {len(puzzle_rows)} is not equal to the needed side length {square_side_length}")


    # Create the goal state
    goal_state = []

    for i in range(0, square_side_length):
        goal_row = []
        for j in range(0, square_side_length):
            goal_row.append(square_side_length * i + (j + 1))
        goal_state.append(goal_row)

    goal_state[square_side_length - 1][square_side_length - 1] = 0

    return puzzle_rows, goal_state

def calculate_state_value_as_manhattan_distance_and_misplaced_tiles(puzzle_rows, goal_state):
    """
        Use sum of manhattan distance and number of misplaced tiles to calculate value of the current state
        In other other words, the sum of the number of squares from the desired location of each title + the number of tiles
        that do not have a value equal to their goal state counterpart.
    """
    
    total_manhattance_distance = 0 
    num_incorrect_tiles = 0

    square_side_length = len(puzzle_rows)

    # Blank tile location 
    blank_tile_location = None

    for i in range(square_side_length):
        for j in range(square_side_length):
            if puzzle_rows[i][j] == 0: 
                blank_tile_location = [i, j] # We don't add blank tile to manhattan distance sum but keep its location for later when generating successor nodes
            else:
                if puzzle_rows[i][j] != goal_state[i][j]:
                    num_incorrect_tiles += 1 

                goal_row = (int(puzzle_rows[i][j]) - 1) // square_side_length
                goal_column = (int(puzzle_rows[i][j]) - 1) % square_side_length

                num_distance = abs(goal_row - i) + abs(goal_column - j)
                total_manhattance_distance += num_distance

    total_cost = total_manhattance_distance + num_incorrect_tiles
                
    return total_cost, blank_tile_location

def generate_successor(square_grid, current_blank_tile_location):
    """
        Generate the successor state by randomly moving the blank tile in one of the directions currently available to it.
        Current blank tile location is a list containing the row, and column of the blank tile respectively.
    """

    square_side_length = len(square_grid)

    # Create a shallow copy of each row to create a deep copy of the original state
    successor_state = [row[:] for row in square_grid]

    # Keep track of possible actions, if blank tile is in top row, can't move up, if all the way on the right, can't move right, etc
    legal_actions = []
    if current_blank_tile_location[0] > 0: # Can move blank tile up since not on top row, 0 is top row
        legal_actions.append((-1, 0))
    if current_blank_tile_location[0] < square_side_length - 1: # Can move blank tile down since not on bottom row, square side length - 1 is bottom row
        legal_actions.append((1, 0)) 
    if current_blank_tile_location[1] > 0: # Can move blank tile to the left since its not all the way on the left
        legal_actions.append((0, -1))
    if current_blank_tile_location[1] < square_side_length - 1: # Not all the way on the right, so we can move blank tile to the right
        legal_actions.append((0, 1))

    # From the available legal moves, randomly choose one
    selected_action = random.choice(legal_actions)

    # New blank tile row
    new_row = current_blank_tile_location[0] + selected_action[0]

    # New blank tile column
    new_column = current_blank_tile_location[1] + selected_action[1]

    # Update the successor's blank tile location to store the number at the location which the blank tile is about to be moved to
    successor_state[current_blank_tile_location[0]][current_blank_tile_location[1]] = successor_state[new_row][new_column]

    # Now move the blank tile to its new location
    successor_state[new_row][new_column] = 0

    return successor_state

def run_simulated_annealing(square_grid, goal_state, scheduler):

    # Starting timer
    algorithm_start_time = time.time()

    # Start with current set to the initial state
    current = square_grid

    # Get value of this state so we can compare it to the succesor, we also get the blank tile location so we can generate its successor
    current_value, current_blank_tile_location = calculate_state_value_as_manhattan_distance_and_misplaced_tiles(square_grid, goal_state)
    
    # Used to track the number of times the current state was updated to the next state
    number_of_actions = 0

    # Start counting from 1 to infinity, t represents time
    for t in itertools.count(1):

        # If current state is already equal to goal state  
        if current_value == 0: 
            break

        # Map from time to temperature for calculating probability of acccepting worse solutions or stopping the program
        temperature = scheduler.get_new_temperature(t)

        # Division by 0 is undefined, stop algorithm and return current. If temperature < 0, we exceeded the min temperature so still stop.
        if temperature <= 0:
            break

        # Generate a successor state by moving the blank tile to a random legal position
        potential_successor = generate_successor(current, current_blank_tile_location)

        # Calculate the value of the successor state, in addition to the new location of the blank tile 
        potential_successor_value, potential_successor_blank_tile_location = calculate_state_value_as_manhattan_distance_and_misplaced_tiles(potential_successor, goal_state)
        
        # This is a minimization problem, so if current - successor value is greater than 0, potential successor's value (cost) is smaller (and therefore better)
        value_increase = current_value - potential_successor_value

        # If new state's value is better, update current to the new state
        if value_increase > 0:
            current = potential_successor
            current_value = potential_successor_value
            current_blank_tile_location = potential_successor_blank_tile_location
            number_of_actions += 1 
        else: # Otherwise do it based on a decreasing probability as temperature gets smaller
            # Do we set current to next even if its worse for the chance of escaping local optima
            set_current_to_sucessor = False

            # Generate a random float between 0.0 and 1.0
            random_num = random.random()

            # Generate probability using the formula given by the SA algorithm
            selection_probablity = math.e ** (value_increase / temperature)

            # If the random number we generated is less than or equal to that probability
            if random_num <= selection_probablity:
                set_current_to_sucessor = True

            # Update current to next, along with its value
            if set_current_to_sucessor:
                current = potential_successor
                current_value = potential_successor_value
                current_blank_tile_location = potential_successor_blank_tile_location
                number_of_actions += 1
        
    # Stop the timer
    algorithm_end_time = time.time()

    # Get the runtime in seconds
    algorithm_run_time = algorithm_end_time - algorithm_start_time

    # Return found solution, it's value, and the time the algorihm took in seconds
    return current, current_value, algorithm_run_time, number_of_actions
    
def main():
    print("Welcome to a Simulated Annealing Solution of the N-puzzle problem.")
    print("To start please provide a value of N. There will be N labeled cells 1 - N, and one blank cell for a total of N + 1 cells.")

    # Store information needed to create the initial grid and goal state
    n_value = None
    square_side_length = None

    invalid = True
    while invalid:
        try: 
            n_value = int(input("Enter your integer input N in the range 8 <= N < 100 where N + 1 is a perfect square: "))
            square_side_length = math.isqrt(n_value + 1)
            if n_value < 8 or n_value >= 100:
                print("Your given N value is not in the allowed range.")
            elif not((square_side_length * square_side_length) == n_value + 1):
                print("Your given N value plus 1 is not a perfect square.")
                print("Note that valid Ns in the given range are 8, 15, 24, 35, 48, 63, 80, and 99.")
            else:
                print(f"Your N is {n_value}")
                print(f"A square grid of size {square_side_length * square_side_length} will be created matching your input file.")
                invalid = False
        except ValueError: 
            print(f"Conversion failed: {n_value} is not an integer")
        except TypeError: 
            print(f"Input type was unrecognized. Please give a valid string representation of an Integer.")

    # Get input, can potentially use different input files depending on the N-value you are trying to use
    print(f"Please provide an N-puzzle problem input representing the start case. Dimensions should be of size {square_side_length} by {square_side_length}")
    file_path = input("Please provide the file path to your input: ")

    # Make sure file exists before we try to read from it
    if os.path.exists(file_path):
        print("File found, creating square grid.")
        square_grid, goal_state = read_puzzle_from_input(file_path, square_side_length, n_value)
    else:
        raise ValueError("The input does not exist at the file path you provided.")

    initial_state_cost, _ = calculate_state_value_as_manhattan_distance_and_misplaced_tiles(square_grid, goal_state)

    print("This program uses a combined heuristic cost of states that is the sum of the manhattance distance of each tile from its desired location and the number of misplaced tiles.")

    # Show how far initial state is from goal state for comparison with SA returned state
    print(f"Your initial state has a combined heuristic cost of {initial_state_cost}, the desired number is 0")
    
    # Allow user to verify their input
    print("Your square grid was read in as follows:")
    for row in square_grid:
        print(row)

    # Print goal state so user can later verify the SA solution
    print("Your goal state grid was calculated as follows:")
    for goal_row in goal_state:
        print(goal_row)

    print("You will now be granted the opportunity to configure some variables for the SA Algorithm.")
    
    temperature = None
    invalid_temperature = True
    while invalid_temperature:
        try:
            temperature = float(input("Please enter your starting temperature (how likely SA will make decisions at the start to escape local maxima: "))
            invalid_temperature = False
        except ValueError: 
            print("Temperature must be a number.")
        except TypeError:
            print("Temperature should be an integer or floating point number.")
    
    invalid_cooling_method = True
    cooling_method = None
    while invalid_cooling_method:
        print("Please choose between a Linear or Geometric Schedule for mapping between time and temperature.")
        print("A linear schedule simply retrieves the current temperature by taking the initial temperature - (current iteration * cooling rate). SA will stop when temperature becomes 0.")
        print("A geometric schedule follows an exponential decay pattern where current temperature is multiplied by a constant factor alpha less than 1 each iteration.")
         
        cooling_method = input("Please enter 'Linear' or 'Geometric' to select your schedule: ")

        if cooling_method != "Linear" and cooling_method != "Geometric":
            print("Option not recognized, please choose a supported scheduler.")
        else:
            invalid_cooling_method = False

    if cooling_method == "Linear":
        cooling_rate = None  
        invalid_cooling_rate = True
        while invalid_cooling_rate:
            try:
                print("Please note that the cooling rate should be greater than 0, and less than or equal to your starting temperature.")
                cooling_rate = float(input("Please enter your cooling rate (how fast temperature decreases per iteration in SA): "))
                
                if cooling_rate <= 0 or cooling_rate > temperature:
                    print(f"Please ensure cooling rate is in the range 0 < x <= {temperature} where x is the cooling rate.")
                else: 
                    invalid_cooling_rate = False
            except ValueError: 
                print("Cooling rate must be a number.")
            except TypeError:
                print("Cooling rate should be an integer or floating point number.")

        # Create a scheduler that will map time to temperature. 
        # For arguments, it accepts the initial starting temperature of the SA algorithm, and the cooling rate.
        # For a cooling rate of 1, and an intial temperature of 10,000, the temperature on the second iteration will be 10,000 - (1 * 2) = 9,998
        # The greater the temperature on an iteration, the higher the chance of the SA algorithm accepting a worse move for that iteration
        scheduler = LinearScheduler(temperature, cooling_rate)
    else:
        alpha = None  
        invalid_alpha = True
        while invalid_alpha:
            try:
                alpha = float(input("Please enter your constant factor alpha where 0 < a < 1: "))
                if alpha <= 0 or alpha >= 1:
                    print("Please ensure alpha is in the allowed range.")
                else: 
                    invalid_alpha = False
            except ValueError: 
                print("Alpha must be a floating point number.")
            except TypeError:
                print("Alpha must be a floating point number.") 
                
        invalid_min_temperature = True
        min_temperature = None
        while invalid_min_temperature:
            try:
                print("The SA algorithm will stop when current temperature goes below the minimum temperature.")
                min_temperature = float(input("Please enter your minimum temperature where minimum temperature > 0: "))
                if min_temperature <= 0:
                    print("Please ensure your minimum temperature is greater than 0")
                else: 
                    invalid_min_temperature = False
            except ValueError: 
                print("Minimum temperature must be a floating point or integer number.")
            except TypeError:
                print("Minimum temperature must be a floating point or integer number.")

        # Create a geometric scheduler that requires starting temperature, a constant factor alpha to reduce current temperature each iteration,
        # and a temperature for which to stop SA when reached.
        scheduler = GeometricScheduler(temperature, alpha, min_temperature)

    best_solution_found, best_solution_cost, algorithm_run_time, number_of_actions = run_simulated_annealing(square_grid, goal_state, scheduler)

    print(f"The algorithm took {algorithm_run_time} seconds to run and updated the current state {number_of_actions} times.")
    print(f"The best solution found has a combined hueristic cost of: {best_solution_cost}")
    print(f"Compared to the initial state, this provides a heuristic cost reduction of {initial_state_cost - best_solution_cost}")
    if best_solution_cost == 0:
        print("Since the final solution's cost is 0, it is equal to the goal state.")
    else:
        print("Since the final solution's cost is not 0, the goal state was not found using SA")
    print("The final state result can be seen as follows")
    for final_solution_row in best_solution_found:
        print(final_solution_row)

if __name__ == "__main__":
    main()
