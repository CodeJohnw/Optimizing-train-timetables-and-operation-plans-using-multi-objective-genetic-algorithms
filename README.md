# Optimizing-train-timetables-and-operation-plans-using-multi-objective-genetic-algorithms

## Overview

This Python script is designed for optimizing train scheduling and operations based on specific parameters of train stations and their respective flows. The script leverages the power of genetic algorithms to optimize train operations, ensuring effective handling of passengers while minimizing operational costs. The script reads multiple datasets, performs complex calculations, and provides a framework to simulate and optimize train loop operations using genetic algorithms.

## Features

- **Data Reading:** Extract data from multiple Excel files to get station data, interval run times, distances, and OD passenger flows.
- **Station Selection:** Identifies and lists stations that can serve as starting and ending points for train loops.
- **Combination Validation:** Ensures that selected start and end stations for train routes meet specific criteria.
- **Optimization Functions:** Includes an objective function to minimize total passenger time (in-vehicle plus waiting time) and operational costs.
- **Constraints Handling:** Incorporates constraints related to train operations and passenger flow management within stations.

## Requirements

- Python 3.x
- `pandas` library for data manipulation and analysis
- `numpy` library for numerical operations
- `matplotlib` for plotting (optional, not used in the script but useful for extensions)
- `sko.GA` from the `scikit-opt` package for running Genetic Algorithm

To install the necessary Python packages, run:
```bash
pip install pandas numpy matplotlib scikit-opt
```

## Usage

1. **Data Preparation:**
   - Ensure that the Excel files (`附件1：车站数据.xlsx`, `附件2：区间运行时间.xlsx`, `附件3：OD客流数据.xlsx`, and `附件4：断面客流数据.xlsx`) are located in the same directory as the script or modify the `file_path` variables in the script to match the locations of these files.

2. **Run Script:**
   - Run the script using a Python environment. Ensure all dependencies are installed.
   - The script will automatically read the data, calculate valid station combinations, and initiate the optimization process.

3. **Output:**
   - The script prints the total number of passengers alighting at each station.
   - Debug and error messages are printed directly to the console to aid in tracing the flow of execution and identifying issues in the constraints or the genetic algorithm parameters.

4. **Optimization:**
   - The genetic algorithm's parameters such as population size, number of generations, mutation rate, etc., can be adjusted in the `GA` function call within the script to experiment with different optimization setups.

## Sample Code Snippet

```python
# Objective function calculation
def objective_function(x):
    # Code to compute the sum of in-vehicle time, waiting time, and operating costs
    return calculated_value

# Main execution point
if __name__ == "__main__":
    ga = GA(func=objective_function, n_dim=155, size_pop=50, max_iter=100, prob_mut=0.01)
    best_x, best_y = ga.run()
    print('Optimal Parameters:', best_x)
    print('Optimal Objective Function Value:', best_y)
```

## Extending the Script

Users can extend this script by:
- Incorporating additional data inputs and constraints.
- Enhancing the objective function to include more detailed aspects of train operation efficiencies.
- Visualizing results using `matplotlib` to graphically represent the optimization outcomes.

For a more customized usage, consider adapting the genetic algorithm parameters or integrating this script into a larger transportation management system.
