import numpy as np
from pyscf import gto, scf, fci, ao2mo
import itertools
import scipy.special
import scipy.sparse

R = 1.1
mol = gto.Mole()
get_atom_Hn = lambda n: [['H', [R * i, 0.0, 0.0]] for i in range(n)]
mol.build(
    atom=get_atom_Hn(4),
    basis='6-31g',
    symmetry=True,
)

hf = scf.RHF(mol)
hf.kernel()
hf.mo_coeff = np.array(hf.mo_coeff)
cisolver = fci.FCI(mol, hf.mo_coeff)
EFCI = cisolver.kernel(davidson_only=False)[0]
print('pyscf energy', EFCI - mol.energy_nuc())


def AO2MO_1e(O, mo_coeff):
    return mo_coeff.T.dot(O).dot(mo_coeff)


h1 = AO2MO_1e(hf.get_hcore(), hf.mo_coeff)
h2 = ao2mo.kernel(mol, hf.mo_coeff)
h2 = ao2mo.restore(1, h2, mol.nao_nr())
nocc = mol.nelectron // 2
nall = mol.nao
nvir = nall - nocc

N = scipy.special.comb(nall, nocc, exact=True)
print('Number of determinants', N**2)


def get_E_J(h2, o1, o2):
    o1 = o1[..., :, None]
    o2 = o2[..., None, :]
    return h2[o1, o1, o2, o2].sum(axis=(-1, -2))


def get_E_K(h2, o1, o2):
    o1 = o1[..., :, None]
    o2 = o2[..., None, :]
    return h2[o1, o2, o2, o1].sum(axis=(-1, -2))


def get_energy_det_0(occ_up, occ_down):
    E_h = h1[occ_up, occ_up].sum(axis=-1) + h1[occ_down, occ_down].sum(axis=-1)
    E_J_aa = get_E_J(h2, occ_up, occ_up) / 2
    E_J_bb = get_E_J(h2, occ_down, occ_down) / 2
    E_J_ab = get_E_J(h2, occ_up, occ_down)
    E_J = E_J_aa + E_J_bb + E_J_ab
    E_K_aa = get_E_K(h2, occ_up, occ_up) / 2
    E_K_bb = get_E_K(h2, occ_down, occ_down) / 2
    E_K = E_K_aa + E_K_bb
    E = E_h + E_J - E_K
    return E


def get_energy_det_1(occ_up, occ_down, p, a, is_up):
    p = np.array(p)[..., None]
    a = np.array(a)[..., None]
    E_h = h1[p, a].squeeze(axis=-1)
    E_J_a = h2[occ_up, occ_up, p, a].sum(axis=-1)
    E_J_b = h2[occ_down, occ_down, p, a].sum(axis=-1)
    if is_up:
        E_K = h2[occ_up, a, p, occ_up].sum(axis=-1)
    else:
        E_K = h2[occ_down, a, p, occ_down].sum(axis=-1)
    return E_h + E_J_a + E_J_b - E_K


def get_energy_det_2(occ_up, occ_down, p, q, a, b, is_up_1, is_up_2):
    # s_p = s_a, s_q = s_b is assumed anyway
    E_J = h2[p, a, q, b]
    E_K = h2[p, b, q, a]
    if is_up_1 == is_up_2:
        return E_J - E_K
    else:
        return E_J


get_energy_det_funcs = {
    0: get_energy_det_0,
    1: get_energy_det_1,
    2: get_energy_det_2,
}


def meshgrid_by_shape(shape):
    return np.meshgrid(*[np.arange(i) for i in shape], indexing='ij')


def get_virs(occs):
    is_occ = np.zeros((N, nall), dtype=bool)
    indices = meshgrid_by_shape((N, nocc))[0]
    is_occ[indices, occs] = True
    is_vir = np.logical_not(is_occ)
    return np.where(is_vir)[1].reshape((N, -1))


def get_combinations_indices(n, k):
    combinations = np.array(list(itertools.combinations(range(n), k)), dtype=int)
    return combinations


def get_ids(occs):
    shape = occs.shape[:-1]
    occs = occs.reshape((-1, nocc))
    ids = np.ravel_multi_index(tuple(occs.T), (nall, ) * nocc)
    return ids.reshape(shape)


def get_id_indices(ids, ids_all):
    shape = ids.shape
    indices = np.searchsorted(ids_all, ids.flatten())
    assert np.all(ids_all[indices] == ids.flatten())
    return indices.reshape(shape)


def get_excite_indices(nocc, nvir, nexcite):
    excite_from = get_combinations_indices(nocc, nexcite)
    Nfrom = len(excite_from)
    excite_to = get_combinations_indices(nvir, nexcite)
    Nto = len(excite_to)
    indices_from = np.repeat(excite_from[:, None, :], Nto, axis=1).reshape((Nfrom * Nto, -1))
    indices_to = np.repeat(excite_to[None, :, :], Nfrom, axis=0).reshape((Nfrom * Nto, -1))
    return indices_from, indices_to


def get_permutation_sign(permutations):
    permutations = permutations.copy()
    n = permutations.shape[-1]
    shape = permutations.shape[0:-1]
    permutations = permutations.reshape((-1, n))
    N = len(permutations)
    nswap = np.zeros((N, ), dtype=int)
    while True:
        finished = True
        for i in range(n - 1):
            is_swap = permutations[np.arange(N), i] > permutations[np.arange(N), i + 1]
            nswap[is_swap] += 1
            permutations[is_swap, i], permutations[is_swap, i + 1] = permutations[is_swap, i + 1], permutations[is_swap, i]
            if np.any(is_swap):
                finished = False
        if finished:
            break
    sign = 1 - (nswap % 2) * 2
    return sign.reshape(shape)


occs = get_combinations_indices(nall, nocc)
virs = get_virs(occs)
ids_occs_all = get_ids(occs)
assert np.all(np.sort(ids_occs_all) == ids_occs_all)
occs_pyscf = (nall - occs - 1)[::-1, ::-1]
order_pyscf = np.argsort(get_ids(occs_pyscf))
order2_pyscf = (order_pyscf[:, None] * N + order_pyscf).flatten()

single_spin_excitation_configurations = {}
for nexcite in [0, 1, 2]:
    if nexcite > min(nocc, nvir):
        continue
    indices_excite_from, indices_excite_to = get_excite_indices(nocc, nvir, nexcite)
    N_excite = len(indices_excite_from)
    occs_excite = np.repeat(occs[:, None, :], N_excite, axis=1)
    orbs_excite_from = occs[:, indices_excite_from]
    orbs_excite_to = virs[:, indices_excite_to]
    indices_tmp = meshgrid_by_shape((N_excite, nexcite))[0]
    occs_excite[:, indices_tmp, indices_excite_from] = orbs_excite_to
    permutation_occs_excite = np.argsort(occs_excite, axis=-1)
    ids_occs_excite = get_ids(np.sort(occs_excite, axis=-1))
    indices_occs_excite = get_id_indices(ids_occs_excite, ids_occs_all)
    signs_occs_excite = get_permutation_sign(permutation_occs_excite)
    single_spin_excitation_configurations[nexcite] = {
        'orbs_excite': {
            'from': orbs_excite_from,
            'to': orbs_excite_to,
        },
        'indices_occs_excite': indices_occs_excite,
        'signs_occs_excite': signs_occs_excite,
    }
slic_all = slice(None)
shape_up2full = (slic_all, None, slic_all, None)
shape_down2full = (None, slic_all, None, slic_all)


def get_excitation_arguments(nexcite_up, nexcite_down, typ):
    assert typ in ['from', 'to']
    orbs_excite_up = single_spin_excitation_configurations[nexcite_up]['orbs_excite'][typ]
    orbs_excite_down = single_spin_excitation_configurations[nexcite_down]['orbs_excite'][typ]
    args_up = tuple(np.moveaxis(orbs_excite_up[shape_up2full], -1, 0))
    args_down = tuple(np.moveaxis(orbs_excite_down[shape_down2full], -1, 0))
    return args_up + args_down


forward_dict = {}
for nexcite_up in single_spin_excitation_configurations:
    for nexcite_down in single_spin_excitation_configurations:
        nexcite_tot = nexcite_up + nexcite_down
        if nexcite_tot > 2:
            continue
        occs_up_full = occs[:, None][shape_up2full]
        occs_down_full = occs[:, None][shape_down2full]
        sign_up = single_spin_excitation_configurations[nexcite_up]['signs_occs_excite']
        sign_down = single_spin_excitation_configurations[nexcite_down]['signs_occs_excite']
        args_from = get_excitation_arguments(nexcite_up, nexcite_down, 'from')
        args_to = get_excitation_arguments(nexcite_up, nexcite_down, 'to')
        args_is_up = (True, ) * nexcite_up + (False, ) * nexcite_down
        get_energy_det = get_energy_det_funcs[nexcite_tot]
        values = get_energy_det(occs_up_full, occs_down_full, *args_from, *args_to, *args_is_up)
        values = values * sign_up[shape_up2full] * sign_down[shape_down2full]
        indices_excite_up = single_spin_excitation_configurations[nexcite_up]['indices_occs_excite']
        indices_excite_down = single_spin_excitation_configurations[nexcite_down]['indices_occs_excite']
        forward_dict[(nexcite_up, nexcite_down)] = (indices_excite_up, indices_excite_down, values)


def get_full_FCImatrix(forward_dict):
    Hamitonian = np.zeros((N, N, N, N))
    for key in forward_dict:
        indices_excite_up, indices_excite_down, values = forward_dict[key]
        a, b, _, _ = meshgrid_by_shape(values.shape)
        Hamitonian[a, b, indices_excite_up[shape_up2full], indices_excite_down[shape_down2full]] = values
    Hamitonian_matrix = Hamitonian.reshape((N**2, N**2))
    return Hamitonian_matrix


def FCI_MV(vector):
    vector = vector.reshape((N, N))
    result = np.zeros((N, N))
    for key in forward_dict:
        indices_excite_up, indices_excite_down, values = forward_dict[key]
        indices_excite_up_full = indices_excite_up[shape_up2full]
        indices_excite_down_full = indices_excite_down[shape_down2full]
        result += np.sum(vector[indices_excite_up_full, indices_excite_down_full] * values, axis=(-1, -2))
    return result.flatten()


Hamitonian_matrix = get_full_FCImatrix(forward_dict)
E_direct = np.linalg.eigvalsh(Hamitonian_matrix)[0]
print('direct solver energy', E_direct)
MV = scipy.sparse.linalg.LinearOperator((N**2, N**2), matvec=FCI_MV)
E_iterative = scipy.sparse.linalg.eigsh(MV, k=1)[0][0]
print('iterative solver energy', E_iterative)

# explicit FCI Hamiltoninan from pyscf
#H = fci.direct_spin1.pspace(h1, h2, nall, (nocc, nocc), np=N**2 * 4)[1]
#print(np.linalg.eigh(H)[0][0])
#err_H = np.linalg.norm(H[order2_pyscf][:, order2_pyscf] - Hamitonian_matrix)
#print(err_H)
