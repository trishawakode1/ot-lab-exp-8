"""python -m unittest test_track_b -v"""
import unittest, itertools, numpy as np
import knapsack as K, scheduling as S, qubo as Q

class TestKnapsack(unittest.TestCase):
    def test_exact_solvers_agree_with_brute_force(self):
        for seed in range(40):
            v, w, cap = K.make_instance(12, "uncorrelated" if seed % 2 else "correlated", seed)
            bf, _ = K.brute_force(v, w, cap); d, dm = K.dp(v, w, cap); b, bm, _, fin = K.branch_and_bound(v, w, cap); i, im, ok = K.ilp(v, w, cap)
            self.assertTrue(fin and ok); self.assertEqual({d, b, i}, {bf})
            for m in (dm, bm, im): self.assertLessEqual(w[m].sum(), cap); self.assertEqual(v[m].sum(), bf)   # feasible & value matches
    def test_greedy_feasible_and_half_approx(self):
        for seed in range(40):
            v, w, cap = K.make_instance(15, "uncorrelated", seed); opt = K.dp(v, w, cap)[0]
            g, gm = K.greedy(v, w, cap); g2, gm2 = K.greedy(v, w, cap, best_single=True)
            self.assertLessEqual(w[gm].sum(), cap); self.assertLessEqual(g, opt); self.assertGreaterEqual(2 * g2, opt)
    def test_greedy_can_be_arbitrarily_bad_without_best_single(self):
        v, w = np.array([2, 1000]), np.array([1, 1000]); g, _ = K.greedy(v, w, 1000); g2, _ = K.greedy(v, w, 1000, True)
        self.assertEqual(g, 2); self.assertEqual(g2, 1000)       # ratio-greedy gets 0.2% of optimum

class TestScheduling(unittest.TestCase):
    def test_lpt_worst_case_instance(self):
        p = np.array([3, 3, 2, 2, 2]); self.assertEqual(S.lpt(p, 2)[0], 7); self.assertEqual(S.brute_force(p, 2), 6); self.assertEqual(S.ilp_makespan(p, 2)[0], 6)
    def test_ilp_matches_brute_force_and_lpt_bound(self):
        rng = np.random.default_rng(0)
        for _ in range(25):
            p = rng.integers(1, 21, 8).astype(float); m = 3; opt = S.brute_force(p, m); ilp, ok = S.ilp_makespan(p, m)
            self.assertTrue(ok); self.assertAlmostEqual(ilp, opt); self.assertLessEqual(S.lpt(p, m)[0] / opt, 4 / 3 - 1 / (3 * m) + 1e-9)

class TestQubo(unittest.TestCase):
    def test_ising_conversion_exact_for_all_states(self):
        rng = np.random.default_rng(1); n = 7; M = rng.normal(size=(n, n)); qb = Q.QUBO(M, offset=1.5); J, h, c = qb.to_ising()
        for bits in itertools.product([0, 1], repeat=n):
            x = np.array(bits); s = 1 - 2 * x
            self.assertAlmostEqual(qb.energy(x), Q.QUBO.ising_energy(J, h, c, s), places=9)
    def test_maxcut_energy_is_minus_cut(self):
        edges = [(0, 1, 1.0), (1, 2, 2.0), (2, 3, 1.0), (0, 3, 3.0), (0, 2, 0.5)]; qb = Q.maxcut_qubo(4, edges)
        for bits in itertools.product([0, 1], repeat=4):
            cut = sum(w for i, j, w in edges if bits[i] != bits[j]); self.assertAlmostEqual(qb.energy(bits), -cut)
    def test_cardinality_penalty_zero_iff_k_selected(self):
        n, k = 6, 2; pen = Q.cardinality_penalty(n, k, 3.0)
        for bits in itertools.product([0, 1], repeat=n): self.assertAlmostEqual(pen.energy(bits), 3.0 * (sum(bits) - k) ** 2)
    def test_brute_force_and_energies_all_consistent(self):
        rng = np.random.default_rng(2); qb = Q.QUBO(rng.normal(size=(8, 8))); e, x = qb.brute_force()
        self.assertAlmostEqual(e, qb.energy(x)); self.assertAlmostEqual(e, min(qb.energy(b) for b in itertools.product([0, 1], repeat=8)))
    def test_flip_delta_formula(self):
        rng = np.random.default_rng(3); qb = Q.QUBO(rng.normal(size=(6, 6))); x = rng.integers(0, 2, 6)
        for i in range(6):
            d = (1 - 2 * x[i]) * (qb.M[i, i] + 2 * (qb.M[i] @ x - qb.M[i, i] * x[i])); y = x.copy(); y[i] ^= 1
            self.assertAlmostEqual(qb.energy(y) - qb.energy(x), d)

class TestSwapSA(unittest.TestCase):
    def test_swap_annealing_keeps_cardinality_and_finds_optimum_small(self):
        rng = np.random.default_rng(5); n, k = 8, 3; qb = Q.QUBO(rng.normal(size=(n, n)))
        E = qb.energies_all(); pop = np.array([bin(i).count("1") for i in range(1 << n)]); ref = E[pop == k].min()
        e, x = Q.swap_annealing(qb, np.random.default_rng(0), k, sweeps=60); self.assertEqual(int(x.sum()), k); self.assertAlmostEqual(e, qb.energy(x)); self.assertAlmostEqual(e, ref)

if __name__ == "__main__": unittest.main()
