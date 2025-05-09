import numpy as np
from scipy.spatial import cKDTree
from numba import cuda, prange, njit, jit
from organisms_cuda import _move_kernel

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
        self.species = np.zeros(0, dtype=np.int8)
        self.size = np.zeros(0, dtype=np.float32)
        self.camouflage = np.zeros(0, dtype=np.float32)
        self.defense = np.zeros(0, dtype=np.float32)
        self.attack = np.zeros(0, dtype=np.float32)
        self.vision = np.zeros(0, dtype=np.float32)

        self.metabolism_rate = np.zeros(0, dtype=np.float32)
        self.nutrient_efficiency = np.zeros(0, dtype=np.float32)
        self.diet_type = np.zeros(0, dtype=np.int8)

        self.fertility_rate = np.zeros(0, dtype=np.float32)
        self.offspring_count = np.zeros(0, dtype=np.int32)
        self.reproduction_type = np.zeros(0, dtype=np.int8)

        self.pack_behavior = np.zeros(0, dtype=np.bool_)
        self.symbiotic = np.zeros(0, dtype=np.bool_)

        self.swim = np.zeros(0, dtype=np.bool_)
        self.walk = np.zeros(0, dtype=np.bool_)
        self.fly = np.zeros(0, dtype=np.bool_)
        self.speed = np.zeros(0, dtype=np.float32)

        self.energy = np.zeros(0, dtype=np.float32)
        self.x_pos = np.zeros(0, dtype=np.float32)
        self.y_pos = np.zeros(0, dtype=np.float32)

        self._pos_tree = None
        self._env = env
        self._dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], np.float32)
        self._gene_pool = None

    def load_genes(self, gene_pool):
        """
        Load gene ranges/options from the provided gene_pool dict (as returned
        by load_genes_from_file). Builds integer mappings for categorical genes.
        """
        self._gene_pool = gene_pool

        # —— Diet type mapping ——
        # e.g. gene_pool['diet_type'] == ["Herb","Omni","Carn","Photo","Parasite"]
        if 'diet_type' in gene_pool:
            self._diet_options = gene_pool['diet_type']
            # map each category string to a small integer code
            self._diet_map = {opt: idx for idx, opt in enumerate(self._diet_options)}
        else:
            self._diet_options = []
            self._diet_map = {}

        # —— Reproduction type mapping ——
        # e.g. gene_pool['reproduction_type'] == ["Sexual","Asexual"]
        if 'reproduction_type' in gene_pool:
            self._repro_options = gene_pool['reproduction_type']
            self._repro_map = {opt: idx for idx, opt in enumerate(self._repro_options)}
        else:
            self._repro_options = []
            self._repro_map = {}


    # Get methods
    def get_organisms(self):
        return self._organisms

    def build_spatial_index(self):
        """
        Build or rebuild the KD-Tree index over organism positions.
        Call this once per tick (after any moves/spawns) to enable fast
        radius or nearest-neighbor queries via self._pos_tree.
        """
        # if we have any organisms, stack their x/y into an (N,2) array…
        if self.x_pos.size > 0:
            coords = np.stack((self.x_pos, self.y_pos), axis=1)

            # cKDTree is much faster for large N
            self._pos_tree = cKDTree(coords)
        else:
            # no points → no tree
            self._pos_tree = None

    def spawn_initial_organisms(
        self,
        number_of_organisms: int,
        randomize: bool = False
    ) -> int:
        """
        Spawns up to `number_of_organisms` with random traits (or defaults),
        filters out any that land out of bounds or on the wrong terrain,
        appends them to the existing arrays, and returns how many were placed.
        """
        import numpy as np

        # Environment info
        w = self._env.get_width()
        h = self._env.get_length()
        terrain = self._env.get_terrain()  # assume shape (h, w)

        n = number_of_organisms

        # Generate trait arrays of length n
        # ─── categorical codes ───
        species_arr = np.zeros(n, dtype=np.int8)              # all same species=0
        diet_arr    = np.full(n, self._diet_map["Herb"], dtype=np.int8)
        repro_arr   = np.full(n, self._repro_map["Asexual"],    dtype=np.int8)

        # ─── continuous traits ───
        if randomize:
            size_arr               = np.random.rand(n).astype(np.float32)
            camouflage_arr         = np.random.rand(n).astype(np.float32)
            defense_arr            = np.random.rand(n).astype(np.float32)
            attack_arr             = np.random.rand(n).astype(np.float32)
            vision_arr             = np.random.rand(n).astype(np.float32)
            metabolism_rate_arr    = np.random.rand(n).astype(np.float32)
            nutrient_efficiency_arr= np.random.rand(n).astype(np.float32)
            fertility_rate_arr     = np.random.rand(n).astype(np.float32)
            offspring_count_arr    = np.random.randint(1, 5, size=n).astype(np.int32)
            speed_arr              = np.random.uniform(0.1, 5.0, size=n).astype(np.float32)
            energy_arr             = np.random.uniform(10, 20, size=n).astype(np.float32)
        else:
            size_arr               = np.full(n, 1.0, dtype=np.float32)
            camouflage_arr         = np.zeros(n, dtype=np.float32)
            defense_arr            = np.zeros(n, dtype=np.float32)
            attack_arr             = np.zeros(n, dtype=np.float32)
            vision_arr             = np.full(n, 15.0, dtype=np.float32)
            metabolism_rate_arr    = np.full(n, 1.0, dtype=np.float32)
            nutrient_efficiency_arr= np.full(n, 1.0, dtype=np.float32)
            fertility_rate_arr     = np.full(n, 0.1, dtype=np.float32)
            offspring_count_arr    = np.full(n, 1, dtype=np.int32)
            speed_arr              = np.full(n, 1.0, dtype=np.float32)
            energy_arr             = np.full(n, 20.0, dtype=np.float32)

        # ─── boolean traits ───
        if randomize:
            pack_arr    = np.random.choice([False, True], size=n)
            symbiotic_arr = np.random.choice([False, True], size=n)
            swim_arr    = np.random.choice([False, True], size=n)
            walk_arr    = np.random.choice([False, True], size=n)
            fly_arr     = np.random.choice([False, True], size=n)
        else:
            pack_arr      = np.zeros(n, dtype=bool)
            symbiotic_arr = np.zeros(n, dtype=bool)
            swim_arr      = np.zeros(n, dtype=bool)
            walk_arr      = np.ones(n,  dtype=bool)  # default walkers
            fly_arr       = np.zeros(n, dtype=bool)

        # Sample positions (int coords)
        x_rand = np.random.randint(0, w, size=n)
        y_rand = np.random.randint(0, h, size=n)
        pos    = np.column_stack((x_rand, y_rand))  # shape (n,2)

        # Terrain values at each candidate
        terr_vals = terrain[pos[:,1], pos[:,0]]  # note: [y, x]

        # Build a single Boolean mask of “truly valid” spawns:
        #    flyers always OK, swimmers only where terr<0, walkers only where terr>=0
        valid = (
            fly_arr |
            ((swim_arr) & (terr_vals < 0)) |
            ((walk_arr) & (terr_vals >= 0))
        )

        # Extract only the valid subset
        idx           = np.nonzero(valid)[0]
        valid_count   = idx.shape[0]
        if valid_count == 0:
            return 0

        # Positions to float for storing
        x_new = pos[idx, 0].astype(np.float32)
        y_new = pos[idx, 1].astype(np.float32)

        # Truncate & align all trait arrays
        species_v            = species_arr[idx]
        size_v               = size_arr[idx]
        camouflage_v         = camouflage_arr[idx]
        defense_v            = defense_arr[idx]
        attack_v             = attack_arr[idx]
        vision_v             = vision_arr[idx]
        metabolism_rate_v    = metabolism_rate_arr[idx]
        nutrient_efficiency_v= nutrient_efficiency_arr[idx]
        diet_v               = diet_arr[idx]
        fertility_rate_v     = fertility_rate_arr[idx]
        offspring_count_v    = offspring_count_arr[idx]
        repro_v              = repro_arr[idx]
        pack_v               = pack_arr[idx]
        symbiotic_v          = symbiotic_arr[idx]
        swim_v               = swim_arr[idx]
        walk_v               = walk_arr[idx]
        fly_v                = fly_arr[idx]
        speed_v              = speed_arr[idx]
        energy_v             = energy_arr[idx]

        # Append to your per-field arrays
        self.species            = np.concatenate((self.species,            species_v))
        self.diet_type          = np.concatenate((self.diet_type,          diet_v))
        self.reproduction_type  = np.concatenate((self.reproduction_type,  repro_v))

        self.size               = np.concatenate((self.size,               size_v))
        self.camouflage         = np.concatenate((self.camouflage,         camouflage_v))
        self.defense            = np.concatenate((self.defense,            defense_v))
        self.attack             = np.concatenate((self.attack,             attack_v))
        self.vision             = np.concatenate((self.vision,             vision_v))
        self.metabolism_rate    = np.concatenate((self.metabolism_rate,    metabolism_rate_v))
        self.nutrient_efficiency= np.concatenate((self.nutrient_efficiency,nutrient_efficiency_v))
        self.fertility_rate     = np.concatenate((self.fertility_rate,     fertility_rate_v))
        self.offspring_count    = np.concatenate((self.offspring_count,    offspring_count_v))
        self.speed              = np.concatenate((self.speed,              speed_v))

        self.pack_behavior      = np.concatenate((self.pack_behavior,      pack_v))
        self.symbiotic          = np.concatenate((self.symbiotic,          symbiotic_v))
        self.swim               = np.concatenate((self.swim,               swim_v))
        self.walk               = np.concatenate((self.walk,               walk_v))
        self.fly                = np.concatenate((self.fly,                fly_v))

        self.energy             = np.concatenate((self.energy,             energy_v))
        self.x_pos              = np.concatenate((self.x_pos,              x_new))
        self.y_pos              = np.concatenate((self.y_pos,              y_new))

        # Update the birth counter and return
        self._env.add_births(valid_count)
        return valid_count
    
    def move(self):
        N = self.x_pos.size
        if N == 0:
            return

        # 1) Build spatial index and query neighbors
        self.build_spatial_index()
        coords = np.stack((self.x_pos, self.y_pos), axis=1)
        raw_lists = self._pos_tree.query_ball_point(coords, self.vision)

        # 2) Flatten neighbor lists
        counts = [len(lst) for lst in raw_lists]
        total = sum(counts)
        neigh_ptrs = np.empty(N+1, dtype=np.int32)
        neigh_ptrs[0] = 0
        np.cumsum(counts, out=neigh_ptrs[1:])
        neigh_inds = np.empty(total, dtype=np.int32)
        idx = 0
        for i_idx, lst in enumerate(raw_lists):
            neigh_inds[idx:idx+counts[i_idx]] = lst
            idx += counts[i_idx]

        dir_count = self._dirs.shape[0]
        # 3) Transfer data to GPU
        d_x           = cuda.to_device(self.x_pos)
        d_y           = cuda.to_device(self.y_pos)
        d_e           = cuda.to_device(self.energy)
        d_cam         = cuda.to_device(self.camouflage)
        d_vis         = cuda.to_device(self.vision)
        d_att         = cuda.to_device(self.attack)
        d_def         = cuda.to_device(self.defense)
        d_pack        = cuda.to_device(self.pack_behavior)
        d_swim        = cuda.to_device(self.swim)
        d_walk        = cuda.to_device(self.walk)
        d_fly         = cuda.to_device(self.fly)
        d_speed       = cuda.to_device(self.speed)
        d_spc         = cuda.to_device(self.species)
        d_terrain     = cuda.to_device(self._env.get_terrain())
        terrain_h, terrain_w = self._env.get_length(), self._env.get_width()
        d_dirs        = cuda.to_device(self._dirs)
        d_dirs_count  = cuda.to_device(dir_count)
        d_ptrs        = cuda.to_device(neigh_ptrs)
        d_inds        = cuda.to_device(neigh_inds)

        # Allocate output arrays
        d_new_x = cuda.device_array_like(d_x)
        d_new_y = cuda.device_array_like(d_y)
        d_new_e = cuda.device_array_like(d_e)

        # 4) Launch kernel
        threads_per_block = 128
        blocks_per_grid  = (N + threads_per_block - 1) // threads_per_block
        _move_kernel[blocks_per_grid, threads_per_block](
            d_x, d_y, d_e,
            d_cam, d_vis, d_att, d_def,
            d_pack, d_swim, d_walk, d_fly, d_speed,
            d_spc,
            d_terrain, terrain_w, terrain_h,
            d_dirs, d_dirs_count, d_ptrs, d_inds,
            d_new_x, d_new_y, d_new_e
        )
        cuda.synchronize()

        # 5) Copy back
        self.x_pos[:]   = d_new_x.copy_to_host()
        self.y_pos[:]   = d_new_y.copy_to_host()
        self.energy[:]  = d_new_e.copy_to_host()

    def resolve_attacks(self):
        """
        Vectorized one‐to‐one attacks on nearest neighbor within vision.
        Attackers gain energy equal to the damage they inflict.
        """
        N = orgs.shape[0]
        if N < 2:
            return

        # --- 0) Extract flat arrays once ---
        coords = np.stack((self.x_pos, self.y_pos), axis=1)  # (N,2)
        att = orgs['attack']      # (N,)
        deff = orgs['defense']
        vision = orgs['vision']
        pack = orgs['pack_behavior']
        fly = orgs['fly']
        swim = orgs['swim']
        walk = orgs['walk']
        energy = orgs['energy']
        terrain = self._env.get_terrain()

        # only use pack logic if any pack_behavior is True
        use_pack = bool(pack.any())

        # --- 1) Ensure KD-Tree is fresh ---
        if self._pos_tree is None:
            self.build_spatial_index()

        # --- 2) Batch nearest-neighbor query (k=2) ---
        dists, idxs = self._pos_tree.query(coords, k=2, workers=-1)
        nearest = idxs[:, 1]          # (N,)
        nearest_dist = dists[:, 1]

        # --- 3) Filter to those within vision ---
        can_see = nearest_dist <= vision
        attackers = np.nonzero(can_see)[0]
        if attackers.size == 0:
            return

        i = attackers                 # attacker indices
        j = nearest[attackers]        # corresponding defender indices

        # --- 4) Terrain/fly/swim/walk restrictions ---
        ix = orgs['x_pos'][j].astype(int)
        iy = orgs['y_pos'][j].astype(int)
        tiles = terrain[iy, ix]       # (M,)

        invalid = np.zeros_like(i, dtype=bool)
        invalid |= (~fly[i] & fly[j])
        invalid |= (~swim[i] & swim[j] & (tiles < 0))
        invalid |= (swim[i] & ~walk[i] & (tiles >= 0))
        invalid |= (swim[i] & ~fly[i] & fly[j] & (tiles < 0))

        valid_mask = ~invalid
        i = i[valid_mask]
        j = j[valid_mask]
        if i.size == 0:
            return

        # --- 5) Compute net attack values ---
        my_net = att[i] - deff[j]    # (K,)
        their_net = att[j] - deff[i]

        # --- 6) Classify host vs prey ---
        if use_pack:
            non_pack = pack[i] & ~pack[j]
            host = (their_net > my_net) & non_pack
            prey = (my_net > their_net) & non_pack
        else:
            host = their_net > my_net
            prey = my_net > their_net

        # only positive damage engagements
        host &= (their_net > 0)
        prey &= (my_net > 0)

        # --- 7) Apply damage and energy gain in batch ---
        # Hostiles: neighbor j attacked i
        if np.any(host):
            idx_i = i[host]           # defenders hit
            idx_j = j[host]           # attackers
            dmg = their_net[host]
            energy[idx_i] -= dmg      # defender loses
            energy[idx_j] += dmg      # attacker gains

        # Prey: attacker i hit neighbor j
        if np.any(prey):
            idx_i = i[prey]           # attackers
            idx_j = j[prey]           # defenders hit
            dmg = my_net[prey]
            energy[idx_j] -= dmg      # defender loses
            energy[idx_i] += dmg      # attacker gains

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
        return
