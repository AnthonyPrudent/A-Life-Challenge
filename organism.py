import numpy as np
import random


class Organisms:
    """
    Represents all organisms in an environment.
    Keeps track of all organism's statistics.
    """

    def __init__(self, env: object):
        """
        Initialize an organism object.

        :param env: 2D Simulation Environment object
        """

        self._organism_dtype = np.dtype([
            ('species', np.str_, 15),
            ('size', np.float32),
            ('speed', np.float32),
            ('max_age', np.float32),
            ('energy_capacity', np.float32),
            ('move_eff', np.float32),
            ('reproduction_eff', np.float32),
            ('min_temp_tol', np.float32),
            ('max_temp_tol', np.float32),
            ('energy_prod', np.str_, 15),
            ('move_aff', np.str_, 15),
            ('energy', np.float32),
            ('x_pos', np.float32),
            ('y_pos', np.float32),
        ])

        self._organisms = np.zeros((0,), dtype=self._organism_dtype)
        self._env = env
        # TODO: Load genes from json file
        self._gene_pool = None

    # Get methods
    def get_organisms(self):
        return self._organisms

    # Set methods
    def set_organisms(self, new_organisms):
        self._organisms = new_organisms

    # TODO: Split logic into smaller functions for readability
    def spawn_initial_organisms(self, number_of_organisms: int,
                                randomize: bool = False) -> int:
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

        # Use gene pool to randomize starting organisms if requested
        if randomize:

            # TODO: Randomize the rest of the genes
            speeds = np.random.randint(1, 5,
                                       size=(number_of_organisms,
                                             ).astype(np.float32))

        # All initial organisms start with the same stats
        # TODO: Use gene pool to create default values, currently hard coded
        else:
            species = np.full((number_of_organisms,), "ORG", dtype=np.str_)
            sizes = np.full((number_of_organisms,), 1, dtype=np.float32)
            speeds = np.full((number_of_organisms,), 1, dtype=np.float32)
            max_ages = np.full((number_of_organisms,), 5, dtype=np.float32)
            energy_capacities = np.full((number_of_organisms,),
                                        1.0, dtype=np.float32)
            move_efficiencies = np.full((number_of_organisms,),
                                        0.01, dtype=np.float32)
            reproduction_efficiencies = np.full((number_of_organisms,),
                                                0.1, dtype=np.float32)
            min_temp_tols = np.full((number_of_organisms,),
                                    2, dtype=np.float32)
            max_temp_tols = np.full((number_of_organisms,),
                                    2, dtype=np.float32)
            energy_productions = np.full((number_of_organisms,),
                                         "heterotroph", dtype=np.str_)
            move_affordances = np.full((number_of_organisms,),
                                       "terrestrial", dtype=np.str_)
            energies = np.full((number_of_organisms,), 1.0, dtype=np.float32)

        # Randomize starting positions
        positions = np.random.randint(0, grid_size,
                                      size=(number_of_organisms, 2)
                                      ).astype(np.float32)

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
        speeds = speeds[:valid_count]
        sizes = sizes[:valid_count]
        max_ages = max_ages[:valid_count]
        energy_capacities = energy_capacities[:valid_count]
        move_efficiencies = move_efficiencies[:valid_count]
        reproduction_efficiencies = reproduction_efficiencies[:valid_count]
        min_temp_tols = min_temp_tols[:valid_count]
        max_temp_tols = max_temp_tols[:valid_count]
        energy_productions = energy_productions[:valid_count]
        move_affordances = move_affordances[:valid_count]
        energies = energies[:valid_count]

        # Create array of spawned organisms
        spawned_orgs = np.zeros((valid_count,), dtype=self._organism_dtype)
        spawned_orgs['species'] = species
        spawned_orgs['size'] = sizes
        spawned_orgs['speed'] = speeds
        spawned_orgs['max_age'] = max_ages
        spawned_orgs['energy_capacity'] = energy_capacities
        spawned_orgs['move_eff'] = move_efficiencies
        spawned_orgs['reproduction_eff'] = reproduction_efficiencies
        spawned_orgs['min_temp_tol'] = min_temp_tols
        spawned_orgs['max_temp_tol'] = max_temp_tols
        spawned_orgs['energy_prod'] = energy_productions
        spawned_orgs['move_aff'] = move_affordances
        spawned_orgs['energy'] = energies
        spawned_orgs['x_pos'] = positions[:, 0]
        spawned_orgs['y_pos'] = positions[:, 1]

        # Add new data to existing organisms array
        self._organisms = np.concatenate((self._organisms, spawned_orgs))
        self._env.add_births(self._organisms.shape[0])

    def _has_energy_to_reproduce(self):
        """Checks if organism meets energy requirements to reproduce."""
        return self._organisms['energy'] > self._organisms['reproduction_eff'] * self._organisms['energy_capacity']

    # TODO: Implement mutation
    def asexual_reproduce(self):
        """
        Causes all organisms that can to reproduce.
        Spawns offspring near the parent
        """
        # TODO: Add check for asexual reproduction gene
        can_reproduce = self._has_energy_to_reproduce() # and has asexual reproduction gene

        if np.any(can_reproduce):
            parents = self._organisms[can_reproduce]
            parent_reproduction_costs = (self._organisms['reproduction_eff']
                                         [can_reproduce]
                                         *
                                         self._organisms['energy_capacity']
                                         [can_reproduce])

            # Create offspring with same types of stats
            offspring = np.zeros((parents.shape[0],), dtype=self._organism_dtype)

            # Loop to inherit stat values for each stat type
            for field in self._organism_dtype.names:
                if field not in ('x_pos', 'y_pos', 'energy'):
                    offspring[field] = parents[field]

            # Generate random offset value for spawn location of offspring
            offset = np.random.uniform(-2, 2, size=(parents.shape[0], 2))

            # Set offspring spawn location using the random offset
            offspring['x_pos'] = parents['x_pos'] + offset[:, 0]
            offspring['y_pos'] = parents['y_pos'] + offset[:, 1]

            # Set offspring starting energy
            offspring['energy'] = parent_reproduction_costs

            # Reduce parents' energy based on previously calculated reproduction cost
            self._organisms['energy'][can_reproduce] -= parent_reproduction_costs
            # TODO: Implement way to mutate offspring genes
            self._organisms = np.concatenate((self._organisms, offspring))

            self._env.add_births(offspring.shape[0])

    def sexual_reproduce(self):

        # TODO: Add check for gene for sexual reproduction & mutation function
        can_reproduce = self._has_energy_to_reproduce() # and has sexual reproduction

        if np.any(can_reproduce):
            parents = self._organisms[can_reproduce]
            n = parents.shape[0]

            # At least two parents that meet sexual reproduction requirements
            if n < 2:
                return

            # Extract positions of eligible parents
            positions = np.stack((parents['x_pos'], parents['y_pos']), axis=1)

            # Compute pairwise distance matrix
            dist_matrix = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=1)

            # Mark distances over threshold (or diagonal) as invalid
            dist_threshold = 1.0
            np.fill_diagonal(dist_matrix, np.inf)
            potential_parent_pairs = np.argwhere(dist_matrix < dist_threshold)

            used = set()            # Set of parents that have already reproduced
            offspring_list = []

            # Check that current potential parents have not already reproduced
            for potential_parent1, potential_parent2 in potential_parent_pairs:
                if potential_parent1 in used or potential_parent2 in used:
                    continue

                parent1 = parents[potential_parent1]
                parent2 = parents[potential_parent2]

                # Choose one parent to clone offspring from
                if random.random() < 0.5:
                    parent_to_inherit_from = parent1
                else:
                    parent_to_inherit_from = parent2

                # Create offspring with same types of stats as chosen parent
                offspring = np.zeros((parent_to_inherit_from.shape[0],), dtype=self._organism_dtype)

                # Inherit float stats
                for field in self._organism_dtype.names:
                    if field in ['x_pos', 'y_pos', 'energy']:
                        continue
                    offspring[field] = parent1[field]





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
        # self._organisms['energy'][alive_idx] -= 0.01 * displacement

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
