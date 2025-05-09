import numpy as np
import json
import random
import string


class Organisms:
    """
    Represents all organisms in an environment.
    Keeps track of all organism's statistics.
    """

    def __init__(self, env: object, mutation_rate: float):
        """
        Initialize an organism object.

        :param env: 2D Simulation Environment object
        :param mutation_rate: How often mutations occur during reproduction
        """

        self._organism_dtype = np.dtype([
            # — Species Lable
            ('species',             np.str_, 15),

            # — Morphological Genes
            ('size',                np.float32),
            ('camouflage',          np.float32),
            ('defense',             np.float32),
            ('attack',              np.float32),
            ('vision',              np.float32),

            # — Metabolic Genes
            ('metabolism_rate',     np.float32),
            ('nutrient_efficiency', np.float32),
            ('diet_type',           np.str_, 15),

            # — ReproductionGenes
            ('fertility_rate',      np.float32),
            ('offspring_count',     np.int32),
            ('reproduction_type',   np.str_, 15),

            # — BehavioralGenes
            ('pack_behavior',       np.bool_),
            ('symbiotic',           np.bool_),

            # — LocomotionGenes
            ('swim',                np.bool_),
            ('walk',                np.bool_),
            ('fly',                 np.bool_),
            ('speed',               np.float32),

            # — Simulation bookkeeping
            ('energy',              np.float32),
            ('x_pos',               np.float32),
            ('y_pos',               np.float32),
        ])

        self._organisms = np.zeros((0,), dtype=self._organism_dtype)
        self._env = env
        self._mutation_rate = mutation_rate
        with open("gene_settings.json", "r") as file:
            self._gene_pool = json.load(file)

        self._ancestry = {}
        self._species_count = {}

    # Get methods
    def get_organisms(self):
        return self._organisms

    def get_ancestries(self):
        return self._ancestry

    def get_species_count(self):
        return self._species_count

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

        # All initial organisms start with the same stats
        # TODO: Use gene pool to create default values, currently hard coded
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

        terrain_values = env_terrain[iy, ix]
        swim_only = swimmers & ~walkers & ~flyers
        walk_only = walkers & ~swimmers & ~flyers

        # Flyers can be anywhere
        valid_fly_positions = positions[flyers]

        # Swimmers need water
        valid_swim_positions = positions[swim_only & (terrain_values < 0)]

        # Walkers need land
        valid_walk_positions = positions[walk_only & (terrain_values >= 0)]

        positions = np.concatenate((valid_fly_positions, valid_swim_positions,
                                    valid_walk_positions), axis=0)

        valid_count = positions.shape[0]

        # Cut stat arrays to match valid count of organism spawn positions
        species = species[:valid_count]
        sizes = sizes[:valid_count]
        camos = camos[:valid_count]
        defenses = defenses[:valid_count]
        attacks = attacks[:valid_count]
        visions = visions[:valid_count]
        metabolism_rates = metabolism_rates[:valid_count]
        nutrient_effs = nutrient_effs[:valid_count]
        diet_types = diet_types[:valid_count]
        fertility_rates = fertility_rates[:valid_count]
        offspring_counts = offspring_counts[:valid_count]
        reproduction_types = reproduction_types[:valid_count]
        pack_behaviors = pack_behaviors[:valid_count]
        symbiostic_behaviors = symbiostic_behaviors[:valid_count]
        swimmers = swimmers[:valid_count]
        walkers = walkers[:valid_count]
        flyers = flyers[:valid_count]
        speeds = speeds[:valid_count]
        energies = energies[:valid_count]

        # Create array of spawned organisms
        spawned_orgs = np.zeros((valid_count,), dtype=self._organism_dtype)

        if randomize:
            spawned_orgs = self.randomize_genes(spawned_orgs)

        else:
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
            spawned_orgs['reproduction_type'] = reproduction_types
            spawned_orgs['pack_behavior'] = pack_behaviors
            spawned_orgs['symbiotic'] = symbiostic_behaviors
            spawned_orgs['swim'] = swimmers
            spawned_orgs['walk'] = walkers
            spawned_orgs['fly'] = flyers
            spawned_orgs['speed'] = speeds

        spawned_orgs['energy'] = energies
        spawned_orgs['x_pos'] = positions[:, 0]
        spawned_orgs['y_pos'] = positions[:, 1]

        # Add new data to existing organisms array
        self._organisms = np.concatenate((self._organisms, spawned_orgs))
        self._env.add_births(self._organisms.shape[0])
        self._ancestry[species[0]] = []
        self._species_count[species[0]] = self._organisms.shape[0]

    # TODO: Implement mutation and
    #       eventually different sexual reproduction types
    def reproduce(self):
        """
        Causes all organisms that can to reproduce.
        Spawns offspring near the parent
        """

        # Obtains an array of all reproducing organisms
        reproducing = (self._organisms['energy']
                       >
                       self._organisms['fertility_rate']
                       *
                       self._organisms['size'])

        if np.any(reproducing):

            parents = self._organisms[reproducing]
            parent_reproduction_costs = (self._organisms['fertility_rate']
                                         [reproducing]
                                         *
                                         self._organisms['size']
                                         [reproducing])

            # TODO: Implement number of children, currently just one offspring
            # Put children randomly nearby
            offset = np.random.uniform(-2, 2, size=(parents.shape[0], 2))
            offspring = np.zeros((parents.shape[0],),
                                 dtype=self._organism_dtype)

            # Replicate genes from parent into offspring
            offspring = self.replicate_genes(offspring, parents)

            # Create offspring simulation bookkeeping
            offspring['energy'] = parent_reproduction_costs
            self._organisms['energy'][reproducing] -= parent_reproduction_costs
            offspring['x_pos'] = parents['x_pos'] + offset[:, 0]
            offspring['y_pos'] = parents['y_pos'] + offset[:, 1]

            # TODO: Possible to enhance this?
            # Handles speciation and lineage tracking
            for i in range(offspring.shape[0]):
                child = offspring['species'][i]
                parent = parents['species'][i]

                if parent == child:
                    self._species_count[parent] += 1

                else:
                    self._species_count[child] = 1
                    self._ancestry[child] = self._ancestry[parent].copy()
                    self._ancestry[child].append(parent)

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

    def replicate_genes(self, offspring, parents):
        """
        Replicate the genes of the parents onto the offspring,
        applying mutation to genes and speciation

        :param offspring: NumPy array of organism data type
        :param parents: NumPy array of organism data type
        :return offspring: Offspring organism NumPy array
        """
        mutated = np.random.rand(
            len(self._organisms)
            ) < self._mutation_rate

        # Randomize genes for mutated organisms
        mutated_offspring = offspring[mutated]
        offspring[mutated] = self.randomize_genes(mutated_offspring)

        # Non mutated offspring have same genes as parent
        offspring[~mutated] = parents[~mutated]

        return offspring

    def randomize_genes(self, organisms):
        """
        Randmizes the genes in each organism

        :param organisms: NumPy array of organism datatypes
        """
        count = organisms.shape[0]

        # TODO: Incorporate master gene pool
        DIET_TYPES = ['Herb', 'Omni', 'Carn', 'Photo', 'Parasite']
        REPRODUCTION_TYPES = ["Sexual", "Asexual"]

        # TODO: Use AI to change species name of mutated organisms
        species = np.array(random_names(count))
        organisms['species'] = species

        sizes = np.random.uniform(0.0, 1.0,
                                  size=(count,).astype(np.float32))
        organisms['size'] = sizes

        camos = np.random.uniform(0.0, 1.0,
                                  size=(count,).astype(np.float32))
        organisms['camoflauge'] = camos

        defenses = np.random.uniform(0.0, 1.0,
                                     size=(count,).astype(np.float32))
        organisms['defense'] = defenses

        attacks = np.random.uniform(0.0, 1.0,
                                    size=(count,).astype(np.float32))
        organisms['attack'] = attacks

        visions = np.random.uniform(0.0, 1.0,
                                    size=(count,).astype(np.float32))
        organisms['vision'] = visions

        metabolism_rates = np.random.uniform(0.0, 1.0,
                                             size=(count,)
                                             .astype(np.float32))
        organisms['metabolism_rate'] = metabolism_rates

        nutrient_effs = np.random.uniform(0.0, 1.0,
                                          size=(count,)
                                          .astype(np.float32))
        organisms['nutrient_efficiency'] = nutrient_effs

        diets = np.random.choice(DIET_TYPES, size=count, replace=True)
        organisms['diet_type'] = diets

        fertility_rates = np.random.uniform(0.0, 1.0,
                                            size=(count,)
                                            .astype(np.float32))
        organisms['fertility_rate'] = fertility_rates

        offspring_counts = np.random.randint(0, 11,
                                             size=(count,)
                                             .astype(np.float32))
        organisms['offspring_count'] = offspring_counts

        reproductions = np.random.choice(REPRODUCTION_TYPES,
                                         size=count, replace=True)
        organisms['reproduction_type'] = reproductions

        pack_behaviors = np.random.rand(count) < 0.5
        organisms['pack_behavior'] = pack_behaviors

        organisms['symbiotic'] = np.random.rand(count) < 0.5
        organisms['swim'] = np.random.rand(count) < 0.5
        organisms['walk'] = np.random.rand(count) < 0.5
        organisms['fly'] = np.random.rand(count) < 0.5

        speeds = np.random.uniform(0.0, 1.0,
                                   size=(count,).astype(np.float32))
        organisms['speed'] = speeds

        return organisms

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


def random_names(number_of_names):

    names = []

    for i in range(number_of_names):
        random_string = ''.join(random.choices(
            string.ascii_letters + string.digits, k=15))

        names.append(random_string)

    return names
