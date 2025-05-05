import numpy as np


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

        self._food_dtype = np.dtype([
            ('energy', np.float32),
            ('consumed', np.bool),
            ('x_pos', np.float32),
            ('y_pos', np.float32),
        ])
        self._food = np.zeros((0), dtype=self._food_dtype)
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
            energies = np.full((number_of_organisms,), 0.5, dtype=np.float32)

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

    # TODO: Implement mutation and
    #       eventually different sexual reproduction types
    def reproduce(self):
        """
        Causes all organisms that can to reproduce.
        Spawns offspring near the parent
        """

        # Obtains an array of all reproducing organisms
        reproducing = (self._organisms['energy'] >
                       self._organisms['reproduction_eff']
                       * self._organisms['energy_capacity'])

        if np.any(reproducing):

            parents = self._organisms[reproducing]
            parent_reproduction_costs = (self._organisms['reproduction_eff']
                                         [reproducing]
                                         *
                                         self._organisms['energy_capacity']
                                         [reproducing])

            # Put children randomly nearby
            offset = np.random.uniform(-2, 2, size=(parents.shape[0], 2))
            offspring = np.zeros((parents.shape[0],),
                                 dtype=self._organism_dtype)
            offspring['x_pos'] = parents['x_pos'] + offset[:, 0]
            offspring['y_pos'] = parents['y_pos'] + offset[:, 1]

            # Create offspring stats
            offspring['energy'] = parent_reproduction_costs
            self._organisms['energy'][reproducing] -= parent_reproduction_costs
            # TODO: Implement way to mutate offspring genes
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
    
    def get_food(self):
        return self._food

    def remove_dead(self):
        """
        Removes dead organisms from the environment
        """

        #  Retrieves which organisms are dead and updates death counter
        dead_mask = (self._organisms['energy'] <= 0)
        n_dead   = np.count_nonzero(dead_mask)

        #  Update the environment's death tally.
        self._env.add_deaths(n_dead)

        if n_dead > 0:
            #  Pull out the dead organisms all at once.
            dead_orgs = self._organisms[dead_mask]

            #  Build an array of new food items.
            #  One entry per dead organism.
            new_food = np.zeros((n_dead,), dtype=self._food_dtype)

            #  Use each organism’s size as its food energy.
            #  Fallback to 0.0 if no org_ref is present.
            new_food['energy']   = dead_orgs['energy_capacity']
            new_food['consumed'] = False
            new_food['x_pos']    = dead_orgs['x_pos']
            new_food['y_pos']    = dead_orgs['y_pos']

            # Append into your food array
            self._food = np.concatenate((self._food, new_food))

        # Finally, remove the dead from the organism list.
        self._organisms = self._organisms[~dead_mask]