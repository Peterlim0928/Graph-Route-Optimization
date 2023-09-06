# Graph-Route-Optimization

Graph-based route optimization with carpooling considerations for faster travel.

## Description

The project focuses on finding the optimal route from point A to point B within a network of roads, incorporating two route options:

- Standard Route: Takes more time.
- Carpool Route: Faster but requires the possibility of picking up a passenger en route.

## Usage

To use this project, follow these steps:

1. Import the function into your Python script or environment:
```from routing_optimizer import optimalRoute```

2. Define the necessary input variables:
    - start: Starting point as an integer.
    - end: Destination point as an integer.
    - passengers: A list of integers representing passenger locations.
    - roads: A list of tuples containing road information in the format (start_vertex, end_vertex, standard_time, carpool_time).  
    Example input:
    ```
    start = 0
    end = 5
    passengers = [2, 1]
    roads = [
        (4, 5, 200, 2),
        (0, 2, 2, 2),
        (1, 3, 10, 5),
        (3, 5, 50, 50),
        (2, 4, 10, 10),
        (0, 1, 1, 1)
    ]
    ```

3. Call the optimalRoute function with the defined input variables, the result will be a list of integers representing all the vertices visited in the optimal route.

Example output:
```
result = [0, 2, 4, 5]
```

## Features

- Graph-based optimization.
- Route selection based on time efficiency.