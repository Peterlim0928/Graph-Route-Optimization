import heapq

def siftup(heap, pos, endpos):
    """
        This method is copied from heapq module, and modified to be used in MinHeap class
        Sink the item at pos until it is at the end of the heap or larger than its children.
        :param heap: The heap
        :param pos: The position of the item to be sifted up
        :param endpos: The position of the last item in the heap
        :complexity: O(log(N)) where N is the number of items in the heap
    """
    startpos = pos
    newitem = heap[pos]
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2*pos + 1    # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos] < heap[rightpos]:
            childpos = rightpos
        # Move the smaller child up.
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2*pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    heapq._siftdown(heap, startpos, pos)

def heapify(array, size):
    """
        This method is copied from heapq module, and modified to be used in MinHeap class
        Transform list into a heap
        :param array: The array to be heapified
        :param size: The size of the array
        :complexity: O(N) where N is the number of items in the array
    """
    n = size
    for i in reversed(range(n//2)):
        siftup(array, i, n)

# A MinHeap class that uses heapq
# This is done the optimize the time and space complexity of the heap
class MinHeap:
    MIN_CAPACITY = 1

    def __init__(self, size):
        """
            Initialize a min heap with a given size
            :param size: the size of the heap
            :complexity: O(N) where N is the size of the heap
        """
        self.heap = [float('inf')] * max(size, self.MIN_CAPACITY)
        self.length = 0

    def __len__(self):
        """
            Return the length of the heap
            :complexity: O(1)
        """
        return self.length
    
    def is_full(self):
        """
            Return True if the heap is full, False otherwise
            :complexity: O(1)
        """
        return self.length == len(self.heap)
    
    def push(self, item):
        """
            Push an item to the heap
            :param item: The item to be pushed
            :complexity: O(log(N))
        """
        if self.is_full():
            raise IndexError("Heap is full")
        
        self.heap[self.length] = item
        self.length += 1
        heapq._siftdown(self.heap, 0, self.length - 1)

    def is_empty(self):
        """
            Return True if the heap is empty, False otherwise
            :complexity: O(1)
        """
        return self.length == 0
    
    def pop(self):
        """
            Pop the minimum item from the heap
            :complexity: O(log(N))
        """
        if self.is_empty():
            raise IndexError("Heap is empty")
        
        self.heap[0], self.heap[self.length - 1] = self.heap[self.length - 1], self.heap[0]
        self.length -= 1
        siftup(self.heap, 0, self.length)
        return self.heap[self.length]


class Edge:
    """ Edge class for graph. """

    def __init__(self, u, v, w):
        """
            Initialize an edge from u to v with weight w
            :param u: The starting vertex
            :param v: The ending vertex
            :param w: The weight of the edge
            :complexity: O(1)
        """
        self.u = u
        self.v = v
        self.w = w

    def __str__(self):
        """
            Return a string representation of the edge
            :complexity: O(1)
        """
        return f"Edge from ({self.u}) to ({self.v}) with weight ({self.w})"


class Vertex:
    """ Vertex class for graph. """
    
    def __init__(self, id):
        """
            Initialize a vertex with id and empty list of edges
            :param id: The id of the vertex
            :complexity: O(1)
        """
        self.id = id
        self.edges = []

        # for traversal
        self.discovered = False
        self.visited = False

        # for calculating distance
        self.distance = 0
        self.previous = None  # backtracking

    def __lt__(self, other):
        """
            Overriding the less than operator for Vertex
            :complexity: O(1)
        """
        return self.distance < other.distance

    def __gt__(self, other):
        """
            Overriding the greater than operator for Vertex
            :complexity: O(1)
        """
        return self.distance > other.distance

    def __str__(self):
        """
            Return a string representation of the vertex
            :complexity: O(E) where E is the number of edges
        """
        out_str = f"Vertex ({self.id}) has edges"
        for edge in self.edges:
            out_str += f" ({edge.u.id} {edge.v.id})"
        return out_str


class Graph:
    """ Graph class. """
    def __init__(self, Vertices):
        """
            Initialize a graph with a list of vertices
            :param Vertices: A list of vertices that represent the graph
            :complexity: O(V) where V is the number of vertices
        """
        self.vertices = [Vertex(Vertices[i]) for i in range(len(Vertices))]

    def reset(self):
        """
            Resets the graph for traversal
            :complexity: O(V) where V is the number of vertices
        """
        for vertex in self.vertices:
            vertex.discovered = False
            vertex.visited = False
            vertex.distance = 0
            vertex.previous = None

    def add_edge(self, u, v, w):
        """
            Add an edge from u to v with weight w
            :param u: The starting vertex
            :param v: The ending vertex
            :param w: The weight of the edge
            :complexity: O(1)
        """
        self.vertices[u].edges.append(Edge(self.vertices[u], self.vertices[v], w))

    def dijkstra(self, source, destination):
        """
        Function description:
            Finds the shortest path from source to destination using Dijkstra's algorithm.
            The implementation of this function is based on FIT2004 Lecture 4 slides.

        Approach description:
            Use BFS to traverse the graph and MinHeap to keep track of the next vertex to visit.

        :Input:
            argv1 source (Vertex): The starting vertex
            argv2 destination (Vertex): The ending vertex
        :Output: A list of vertices that represent the shortest path from source to destination
        :Time Complexity: O(E log V) where E is the number of edges and V is the number of vertices
        :Aux Space Complexity: O(V) where V is the number of vertices
        """

        # Reset all vertices
        # Time Complexity: O(V) where V is the number of vertices
        self.reset()

        # Initialize the MinHeap
        # Time Complexity: O(V log V) where V is the number of vertices
        heap = MinHeap(len(self.vertices))
        heap.push(source)
        source.discovered = True

        # Keep looping until all vertices are discovered
        # Time Complexity: O(E log V) where V is the number of vertices and E is the number of edges
        while heap.length > 0:
            u = heap.pop()
            u.visited = True

            # Found destination, backtrack to base case
            if u == destination:  
                output = [u.id]
                while u.previous != None:
                    u = u.previous
                    output.append(u.id)
                output.reverse()
                return output

            # Loop through all edges of current vertex
            for edge in u.edges:
                v = edge.v
                if not v.discovered: # If vertex is not discovered
                    v.distance = u.distance + edge.w
                    v.previous = u
                    heap.push(v)
                    v.discovered = True
                elif not v.visited: # If vertex is discovered but not visited
                    if v.distance > u.distance + edge.w: # Update distance if new distance is shorter
                        v.distance = u.distance + edge.w
                        v.previous = u
                        heapify(heap.heap, heap.length)

    def bfs(self, source):
        """
        Function for BFS, starting from source
        """
        self.reset()

        output = []
        discovered = [source]  # queue, FIFO
        discovered[0].discovered = True
        while len(discovered) > 0:
            u = discovered.pop(0)
            output.append(u)
            for edge in u.edges:
                v = edge.v
                if not v.discovered:
                    discovered.append(v)
                    v.discovered = True
        return output

    def __str__(self):
        """
            Return a string representation of the graph
            :complexity: O(V + E) where V is the number of vertices and E is the number of edges
        """
        out_str = f"Graph with {len(self.vertices)} vertices\n{str([vertex.id for vertex in self.vertices])}"
        for vertex in self.vertices:
            out_str += f"\nVertex ({vertex.id}) has edges"
            for edge in vertex.edges:
                out_str += f" ({edge.u.id} {edge.v.id})"
        return out_str


def optimalRoute(start, end, passengers, roads):
    """
    Function description:
        Finds the path that takes the least amount of time to travel from start to end

    Approach description:
        This question is the typical shortest path problem with a twist of having passengers which determines the weight (time) of the edges,
        and each of the edges have two weights (time1 and time2) depending on whether the driver is driving alone or with a passenger.

        Since we cannot create two of the same directed edges, we can create a graph with twice the number of vertices as there are locations.
         - The first half of the vertices and edges represent the roads when driving alone.
         - The second half of the vertices and edges represent the roads when driving with a passenger.

        Connect the first half of the vertices to the second half of the vertices with edges of weight 0 at the locations where there are passengers,
        the driver will start off at first half of the vertices where there are only roads for driving alone, once the driver picks up a passenger,
        the driver will travel to the second half of the vertices where there are only roads for driving with a passenger.

        Add an extra edge from the end location of the second half of the vertices to the end location of the first half of the vertices with weight 0,
        this is to ensure that the driver will always travel to the same end location regardless of whether the driver is driving alone or with a passenger.

        Run Dijkstra's algorithm on the graph to find the shortest path from start to end.

    :Input:
        argv1 start (int): The starting location represented as location id
        argv2 end (int): The ending location represented as location id
        argv3 passengers (list[int]): The locations represented as location ids where there are passengers
        argv4 roads (list[tuple[int, int, int, int]]): The roads represented as (start, end, time1, time2) where 
          start is the starting location of the road, 
          end is the ending location of the road, 
          time1 is the time needed to travel from start to end when driving alone, 
          time2 is the time needed to travel from start to end when driving with a passenger

    :Output:
        A list of locations that represent the shortest path from start to end

    :Time Complexity: O(R log L) where R is the number of roads and L is the number of locations
    :Aux Space Complexity: O(R + L) where R is the number of roads and L is the number of locations
    """

    # Any section without complexity analysis is O(1)

    # Calculate the size of the graph
    # Time Complexity: O(R) where R is the number of roads (edges)
    size = 0
    for road in roads:
        size = max(size, road[0], road[1])
    size += 1

    # Generate the graphs and vertices
    # Time Complexity: O(L) where L is the number of locations (vertices)
    # Space Complexity: O(L)
    graph = Graph([i for i in range(size*2)])

    # Add passengers (edges across graph1 and graph2, graph1 is the first half of the graph and graph2 is the second half of the graph)
    # Time Complexity: O(P) where P is the number of passengers
    # Space Complexity: O(P)
    for passenger in passengers:
        graph.add_edge(passenger, passenger+size, 0)

    # Add edges to the graphs
    # Time Complexity: O(R)
    # Space Complexity: O(R)
    for road in roads:
        graph.add_edge(road[0], road[1], road[2])
        graph.add_edge(road[0]+size, road[1]+size, road[3])

    # Add special edge at the end (edge from graph2 back to graph1)
    # Time Complexity: O(1)
    graph.add_edge(end+size, end, 0)

    # Run Dijkstra's algorithm
    # Time Complexity: O(R log L)
    # Space Complexity: O(R + L)
    lst = graph.dijkstra(graph.vertices[start], graph.vertices[end])

    # Remove duplicates
    # Time Complexity: O(L)
    i = 0
    while i < len(lst):
        lst[i] = lst[i] % size
        if i == 0 or lst[i] != lst[i-1]:
            i += 1
        else:
            lst.pop(i)
    return lst

if __name__ == "__main__":
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
    result = [0, 2, 4, 5]
    out = optimalRoute(start, end, passengers, roads)
    print(out)
    # assert out == result, f"Expected {result}, got {out}"