import numpy as np
from genome import Genome


class Organisms:
    """
    Represents all organisms in an environment.
    Keeps track of all organism's statistics.
    """

    def __init__(self, env: object, mutation_rate: float):
        """
        Initialize an organism object.

        :param env: 2D Simulation Environment object
        :param mutation_rate: How often gene mutations occur when reproducing
        """

        self._organism_dtype = np.dtype([
            ('species', np.str_, 15),
            ('x_pos', np.float32),
            ('y_pos', np.float32),
            ('energy', np.float32),
            ('genome', object)
        ])

        self._organisms = np.zeros((0,), dtype=self._organism_dtype)
        self._env = env
        self._mutation_rate = mutation_rate

    # Get methods
    def get_organisms(self):
        return self._organisms

    # Set methods
    def set_organisms(self, new_organisms):
        self._organisms = new_organisms

    # TODO: Split logic into smaller functions for readability
    def spawn_initial_organisms(self, number_of_organisms: int,
                                randomize: bool = True) -> int:
        """
        Spawns the initial organisms in the simulation.
        Organism stats can be randomized if desired.
        Updates the birth counter in the environment.

        :param number_of_organisms: Number of organisms to spawn
        :param randomize: Request to randomize stats of spawned organisms
        """

        # Unpack important values from args
        env_width = self._env.get_width()
        env_length = self._env.get_length()
        env_terrain = self._env.get_terrain()
        # TODO: Change if environment can be NxM too
        grid_size = env_width

        # Randomizes genes of starting organisms
        if randomize:
            # TODO: Randomize the species name of the starting organisms
            species = np.full((number_of_organisms,), "ORG", dtype=np.str_)
            genomes = np.array([Genome(self._mutation_rate) for _
                                in range(number_of_organisms)], dtype=object)

        # TODO: Allow users to define starting organisms

        # Sets starting position and energy
        positions = np.random.randint(0, grid_size,
                                      size=(number_of_organisms, 2)
                                      ).astype(np.float32)

        energies = np.full((number_of_organisms,), 0.5, dtype=np.float32)

        # Clip positions that are out of bound
        positions = positions[
            (positions[:, 0] >= 0) & (positions[:, 0] < env_width) &
            (positions[:, 1] >= 0) & (positions[:, 1] < env_length)
        ]

        # TODO: Ensure that the number requested is spawned rather than
        #       just valid positions
        # Take rows and columns of the positions and verify those on land
        ix = positions[:, 0].astype(np.int32)
        iy = positions[:, 1].astype(np.int32)
        land_filter = env_terrain[iy, ix] >= 0
        positions = positions[land_filter]
        valid_count = positions.shape[0]

        # Cut stat arrays to match valid count of organism spawn positions
        species = species[:valid_count]
        genomes = genomes[:valid_count]
        energies = energies[:valid_count]

        # Create array of spawned organisms
        spawned_orgs = np.zeros((valid_count,), dtype=self._organism_dtype)
        spawned_orgs['species'] = species
        spawned_orgs['x_pos'] = positions[:, 0]
        spawned_orgs['y_pos'] = positions[:, 1]
        spawned_orgs['energy'] = energies
        spawned_orgs['genome'] = genomes

        # Add new data to existing organisms array
        self._organisms = np.concatenate((self._organisms, spawned_orgs))
        self._env.add_births(self._organisms.shape[0])

    # TODO: Implement mutation and
    #       eventually different sexual reproduction types
    def reproduce(self):
        """
        Causes all organisms that can to reproduce.
        Spawns offspring near the parent
        """

        # Extract reproduction information from all organisms
        genomes = self._organisms['genome']

        repro_eff = np.array([
            genome.get_reproduction().fertility_rate for
            genome in genomes
        ], dtype=np.float32)

        size = np.array([
            genome.get_morphological().size for
            genome in genomes
        ], dtype=np.float32)

        # Obtains an array of all reproducing organisms
        reproducing = (self._organisms['energy'] >
                       repro_eff
                       * size)

        if np.any(reproducing):

            parents = self._organisms[reproducing]
            parent_reproduction_costs = (repro_eff
                                         [reproducing]
                                         *
                                         size
                                         [reproducing])

            # Put children randomly nearby
            offset = np.random.uniform(-2, 2, size=(parents.shape[0], 2))
            offspring = np.zeros((parents.shape[0],),
                                 dtype=self._organism_dtype)
            offspring['x_pos'] = parents['x_pos'] + offset[:, 0]
            offspring['y_pos'] = parents['y_pos'] + offset[:, 1]

            # Energy transfer between parent and child
            offspring['energy'] = parent_reproduction_costs
            self._organisms['energy'][reproducing] -= parent_reproduction_costs

            # Replicates the parent genes, applying mutation
            parents_genomes = parents['genome']
            offspring_genomes = np.array([
                genome.replicate() for genome in parents_genomes
            ], dtype=object)
            offspring['genome'] = offspring_genomes

            # Stores the offsprings
            self._organisms = np.concatenate((self.organisms, offspring))
            self._env.add_births(offspring.shape[0])

    # TODO: Add cost to organism movement based on the movement efficiency
    def move(self):
        """
        Moves all organisms randomly.
        """
        alive = (self._organisms['energy'] > 0)
        speed = self._organisms['speed'][:, None]
        jitter_shape = (alive.sum(), 2)
        move_jitter = np.random.uniform(-1, 1, size=jitter_shape) * speed

        new_positions = np.stack(
            (
                self._organisms['x_pos'],
                self._organisms['y_pos']
            ),
            axis=1
            ) + move_jitter

        self.verify_positions(new_positions)

        # Organisms that moved are incurred an energy cost
        # TODO: Implement movement efficiency gene in cost calc
        alive_idx = np.flatnonzero(self._organisms['energy'])
        displacement = np.abs(
            move_jitter[:alive_idx.shape[0]]
            ).sum(axis=1)
        self._organisms['energy'][alive_idx] -= 0.05 * displacement

    # TODO: Cleanup further and add logic for move affordances
    def verify_positions(self, new_positions):
        """
        Verifies the new positions of all living organisms.
        """

        # Checks that organisms are on terrain they can move on
        # Organisms die if not
        # TODO: Implement movement affordance gene checking
        land_mask = self._env.inbounds(new_positions)
        self._organisms[~land_mask] = 0

        # Only valid organism moves are made
        new_x_positions = new_positions[:, 0]
        new_y_positions = new_positions[:, 1]
        self._organisms['x_pos'] = new_x_positions
        self._organisms['y_pos'] = new_y_positions

    # TODO: Cleanup since organisms eat other organisms
    # Once we deal with speciation, organisms will eat plantlike organisms
    # as an example
    def consume_organism(self):
        """
        """
        pass

    # TODO: Add method for organizim decision making
    def take_action(self):
        pass

    def remove_dead(self):
        """
        Removes dead organisms from the environment
        """

        # Retrieves which organisms are dead and updates death counter
        dead_mask = (self._organisms['energy'] <= 0)
        self._env.add_deaths(np.count_nonzero(dead_mask))

        # The dead are removed from the organisms array
        survivors = self._organisms[~dead_mask]
        self._organisms = survivors
