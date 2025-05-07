# A-Life Challenege
# Zhou, Prudent, Hagan
# Gene implementation

import random
# from zipfile import sizeEndCentDir


class Genome:
    def __init__(self, mutation_rate=0.05):
        # TODO: mutation_rate is currently unused, no mutation functionality yet
        """
        Initialize a genome object.

        :param mutation_rate: A float
        """
        # Min/max values for gene attributes (attack, size, etc) with float values - used by random_stat()
        # function located in Gene parent class for initializing random values
        self._min_val = 0.0
        self._max_val = 1.0

        self._mutation_rate = mutation_rate
        self._morphological_genes = MorphologicalGenes(self._min_val, self._max_val)
        self._metabolic_genes = MetabolicGenes(self._min_val, self._max_val)
        self._reproduction_genes = ReproductionGenes(self._min_val, self._max_val)
        self._locomotion_genes = LocomotionGenes(self._min_val, self._max_val)
        self._behavior_genes = BehaviorGenes(self._min_val, self._max_val)

    def get_all_genes(self):
        return self._morphological_genes, self._reproduction_genes, self._locomotion_genes, self._behavior_genes

    def get_mutation_rate(self):
        return self._mutation_rate

    def get_morphological_genes(self):
        return self._morphological_genes

    def get_metabolic_genes(self):
        return self._metabolic_genes

    def get_reproduction_genes(self):
        return self._reproduction_genes

    def get_locomotion_genes(self):
        return self._locomotion_genes

    def get_behavior_genes(self):
        return self._behavior_genes

    def replicate_genes(self):
        """
        Replicates genes during for organism reproduction,
        potentially causing mutations.
        """
        pass
        # TODO: Just commenting this out for now, if genes are going to be attributes
        # TODO: of Genome, we need a different copy method
        """
        all_genes = self.get_all_genes()
        replicated_genome = {}

        for gene in all_genes:
            # Random.random() returns float 0.0..1.0 - 5 percent chance if mutation_rate set to 0.05
            if random.random() < self._mutation_rate:
                replicated_genome[gene] = self._genes[gene]

            else:
                replicated_genome[gene] = self._genes[gene].mutate()

        child_genome = Genome(self._mutation_rate, replicated_genome)

        return child_genome
        """


class Gene:
    """
    Represents a gene of an organism.
    """

    def __init__(self, min_val, max_val):
        """
        Initialize a gene object.

        :param min_val: An integer or float
        :param max_val: An integer or float
        """
        self._min_val = min_val
        self._max_val = max_val

    def get_min_val(self):
        return self._min_val

    def get_max_val(self):
        return self._max_val

    def random_stat(self) -> object:
        """
        Called during initialization of a child gene to randomize
        values of attributes that use floats.
        """
        return round(random.uniform(self._min_val, self._max_val), 1)


class MorphologicalGenes(Gene):
    """
    Defines the energy gathering method of the organism.
    random_stat currently returns a random value between 0.0 and 1.0
    """
    def __init__(self, min_val, max_val, size=None, camouflage=None,
                 defense=None, attack=None, vision=None):
        super().__init__(min_val, max_val)

        self._size = size if size is not None else self.random_stat()
        self._camouflage = camouflage if camouflage is not None else self.random_stat()
        self._defense = defense if defense is not None else self.random_stat()
        self._attack = attack if attack is not None else self.random_stat()
        self._vision = vision if vision is not None else self.random_stat()

    def mutate(self) -> object:
        pass


class MetabolicGenes(Gene):
    """
    Defines movement affordances of organism.
    """
    def __init__(self, min_val, max_val, metabolic_rate=None, diet_type=None, nutrient_efficiency=None):
        super().__init__(min_val, max_val)

        self._metabolic_rate = metabolic_rate if metabolic_rate is not None \
            else self.random_stat()
        self._diet_type = diet_type if diet_type is not None \
            else random.choice(["herbivore", "omnivore", "carnivore", "photosynthesis", "parasite"])
        self._nutrient_efficiency = nutrient_efficiency if nutrient_efficiency is not None \
            else self.random_stat()


class ReproductionGenes(Gene):
    """Define reproduction of organism."""
    def __init__(self, min_val, max_val, reproduction_type=None, reproduction_efficiency=None, num_of_offspring=None):
        super().__init__(min_val, max_val)

        self._reproduction_type = reproduction_type if reproduction_type is not None \
            else random.choice(["Asexual",  "Sexual"])
        self._reproduction_efficiency = reproduction_efficiency if reproduction_efficiency is not None \
            else self.random_stat()
        self._num_of_offspring = num_of_offspring if num_of_offspring is not None \
            else 1

    def mutate(self):
        pass


class LocomotionGenes(Gene):
    """Define types of movement allowed"""
    def __init__(self, min_val, max_val, walk=None, swim=None, fly=None, speed=None):
        super().__init__(min_val, max_val)

        self._walk = walk if walk is not None else random.choice([True, False])
        self._swim = swim if swim is not None else random.choice([True, False])
        self._fly = fly if fly is not None else random.choice([True, False])
        self._speed = speed if speed is not None else self.random_stat()

    def mutate(self):
        pass


class BehaviorGenes(Gene):
    """Define extra behaviors of organism."""
    def __init__(self, min_val, max_val, pack_behavior=None, symbiotic_behavior=None):
        super().__init__(min_val, max_val)

        self._pack_behavior = pack_behavior if pack_behavior is not None else random.choice([True, False])
        self._symbiotic_behavior = symbiotic_behavior if symbiotic_behavior is not None else random.choice([True, False])

    def mutate(self):
        pass
