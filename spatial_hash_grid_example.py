from collections import defaultdict
import numpy as np


#
# Spatial Hash Grid class
# Would be its own module (ex spatial_hash_grid.py)
#
class SpatialHashGrid:
    def __init__(self, cell_size: float, width: int, height: int):
        self.cell_size = cell_size
        self.width = width
        self.height = height
        self.grid = defaultdict(list)
        self.x_array = None
        self.y_array = None

    def _hash_coords(self, x: float, y: float):
        """Hash floating point coordinates based on into a grid cell"""
        return int(x // self.cell_size), int(y // self.cell_size)

    def clear(self):
        """Clear the spatial hash grid"""
        self.grid.clear()

    def insert(self, idx: int, x: float, y: float):
        """Insert an organism into a grid cell using hashed coordinates"""
        cell = self._hash_coords(x, y)
        self.grid[cell].append(idx)

    def populate(self, x_array: np.ndarray, y_array: np.ndarray):
        """Bulk insert all organisms and store coordinate arrays for distance checks."""
        self.clear()
        self.x_array = x_array
        self.y_array = y_array
        for i in range(len(x_array)):
            self.insert(i, x_array[i], y_array[i])

    def query_neighbors(self, x: float, y: float, radius: float) -> list:
        """
        Return indices of all neighbors within `radius` of (x, y),
        using true Euclidean distance filtering.
        """
        if self.x_array is None or self.y_array is None:
            raise RuntimeError("Must call populate() before query_neighbors().")


        cx, cy = self._hash_coords(x, y)                   # Get the hashed coordinates of the organism being queried
        r_cells = int(np.ceil(radius / self.cell_size))    # Get number of cells to search based on radius (vision)
        r2 = radius * radius                               # squared radius

        results = []
        # Loop through each nearby cell
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                cell = (cx + dx, cy + dy)
                if cell in self.grid:                      # Check that the cell exists
                    for idx in self.grid[cell]:
                        dx = self.x_array[idx] - x
                        dy = self.y_array[idx] - y
                        if dx * dx + dy * dy <= r2:        # squared Euclidean distance check
                            results.append(idx)            # if neighbor coords are within radius (vision) store coords

        return results


def adaptive_radius(env_width: float, env_height: float, vision: float = 0.02) -> float:
    """
    Returns an interaction radius proportional to the size of the environment.
    Vision should be a value between 0.0 and 1.0 (determines radius to check for neighbors)
    """
    return vision * max(env_width, env_height)


#
# Example of step method for Environment class using SpatialHashGrid for neighbor detection
#
def step(self):
"""
Steps one generation forward in the simulation.
"""

organisms_array = self._organisms.get_organisms()
grid = SpatialHashGrid(cell_size=5.0, width=self._width, height=self._length)
grid.populate(organisms_array['x_pos'], organisms_array['y_pos'])

# Check for neighbors - vision determines size of circle where 2 organisms would be neighbors
radius = adaptive_radius(self._width, self._length, vision=0.01)

# Loop through organism numpy array, calls query_neighbors (SpatialHashGrid method)
for i in range(len(organisms_array)):
    neighbors = grid.query_neighbors(
        organisms_array['x_pos'][i],
        organisms_array['y_pos'][i],
        radius=radius  # actual distance
    )
    neighbors = [j for j in neighbors if j != i]  # exclude self

    # Print statement for debugging
    if neighbors:
        pass
    print(f"Org {i} has {len(neighbors)} neighbor(s)")

self._organisms.move()

# TODO: Could this be moved to an org method?
self._organisms.remove_dead()

self._generation += 1