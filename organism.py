import numpy as np
import json


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
            # species label
            ('species',           np.str_,   15),

            # — Morphological Genes
            ('size',              np.float32),
            ('camouflage',        np.float32),
            ('defense',           np.float32),
            ('attack',            np.float32),
            ('vision',            np.float32),

            # — Metabolic Genes
            ('metabolism_rate',   np.float32),
            ('nutrient_efficiency', np.float32),
            ('diet_type',         np.str_,   15),

            # — ReproductionGenes
            ('fertility_rate',    np.float32),
            ('offspring_count',   np.int32),
            ('reproduction_type', np.str_,   15),

            # — BehavioralGenes
            ('pack_behavior',     np.bool_),
            ('symbiotic',         np.bool_),

            # — LocomotionGenes
            ('swim',              np.bool_),
            ('walk',              np.bool_),
            ('fly',               np.bool_),
            ('speed',             np.float32),

            # — Simulation bookkeeping
            ('energy',            np.float32),
            ('x_pos',             np.float32),
            ('y_pos',             np.float32),
        ])

        self._organisms = np.zeros((0,), dtype=self._organism_dtype)
        self._env = env
        with open("gene_settings.json", "r") as file:
            self._gene_pool = json.load(file)

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
            species = np.full((number_of_organisms,), "Original Organism",
                              dtype=np.str_)

            # — Morphological Genes
            sizes = np.full((number_of_organisms,), 0.5, dtype=np.float32)
            camos = np.full((number_of_organisms,), 0.5, dtype=np.float32)
            defenses = np.full((number_of_organisms,), 0.5, dtype=np.float32)
            attacks = np.full((number_of_organisms,), 0.5, dtype=np.float32)
            visions = np.full((number_of_organisms,), 0.5, dtype=np.float32)

            # — Metabolic Genes
            metabolism_rates = np.full((number_of_organisms,),
                                       0.5, dtype=np.float32)
            nutrient_effs = np.full((number_of_organisms,),
                                    0.5, dtype=np.float32)
            diet_types = np.full((number_of_organisms,),
                                 "Photo", dtype=np.str_)

            # — ReproductionGenes
            fertility_rates = np.full((number_of_organisms,),
                                      0.5, dtype=np.float32)
            offspring_counts = np.full((number_of_organisms,),
                                       1, dtype=np.int32)
            reproduction_types = np.full((number_of_organisms,),
                                         "Asexual", dtype=np.str_)

            # — BehavioralGenes
            pack_behaviors = np.full((number_of_organisms,),
                                     False, dtype=np.bool_)
            symbiostic_behaviors = np.full((number_of_organisms,),
                                           False, dtype=np.bool_)

            # — LocomotionGenes
            swimmers = np.full((number_of_organisms,),
                               False, dtype=np.bool_)
            walkers = np.full((number_of_organisms,),
                              True, dtype=np.bool_)
            flyers = np.full((number_of_organisms,),
                             False, dtype=np.bool_)
            speeds = np.full((number_of_organisms,), 0.5, dtype=np.float32)

            # — Simulation bookkeeping
            energies = np.full((number_of_organisms,), 0.25, dtype=np.float32)

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
        energies = energies[:valid_count]

        # Create array of spawned organisms
        spawned_orgs = np.zeros((valid_count,), dtype=self._organism_dtype)
        spawned_orgs['species'] = species
        spawned_orgs['size'] = sizes
        spawned_orgs['camoflauge'] = camos
        spawned_orgs['defense'] = defenses
        spawned_orgs['attack'] = attacks
        spawned_orgs['vision'] = visions
        spawned_orgs['metabolism_rate'] = metabolism_rates
        spawned_orgs['nutrient_efficiency'] = nutrient_effs
        spawned_orgs['diet_type'] = diet_types
        spawned_orgs['fertility_rate'] = fertility_rates
        spawned_orgs['offspring_count'] = offspring_counts
        spawned_orgs['pack_behavior'] = pack_behaviors
        spawned_orgs['symbiotic'] = symbiostic_behaviors
        spawned_orgs['swim'] = swimmers
        spawned_orgs['walk'] = walkers
        spawned_orgs['fly'] = flyers
        spawned_orgs['speed'] = speeds
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
