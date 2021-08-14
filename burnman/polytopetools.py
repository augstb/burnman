# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit
# for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.

from __future__ import absolute_import

import numpy as np
from sympy import Matrix
from scipy.optimize import linprog
from scipy.linalg import block_diag

import logging
import cvxpy as cp

from .polytope import MaterialPolytope, independent_row_indices, complete_basis
from .processchemistry import dictionarize_formula, compositional_array
from .processchemistry import site_occupancies_to_strings
from . import CombinedMineral, SolidSolution, Composite


def solution_polytope_from_charge_balance(charges, charge_total,
                                          return_fractions=False):
    """

    """
    n_sites = len(charges)
    all_charges = np.concatenate(charges)
    n_site_elements = len(all_charges)
    equalities = np.empty((n_sites+1, n_site_elements+1))
    equalities[:-1, 0] = -1
    i = 0
    for i_site, site_charges in enumerate(charges):
        equalities[i_site, 1:] = [1 if (j >= i and j < i+len(site_charges))
                                  else 0 for j in range(n_site_elements)]
        i += len(site_charges)

    equalities[-1, 0] = -charge_total
    equalities[-1, 1:] = all_charges

    pos_constraints = np.concatenate((np.zeros((len(equalities[0])-1, 1)),
                                      np.identity(len(equalities[0]) - 1)),
                                     axis=1)
    return MaterialPolytope(equalities, pos_constraints,
                            return_fractions=return_fractions)


def solution_polytope_from_endmember_occupancies(endmember_occupancies,
                                                 return_fractions=False):
    """

    """
    n_sites = sum(endmember_occupancies[0])
    n_occs = endmember_occupancies.shape[1]

    nullspace = np.array(Matrix(endmember_occupancies).nullspace(),
                         dtype=np.float)

    equalities = np.zeros((len(nullspace)+1, n_occs+1))
    equalities[0, 0] = -n_sites
    equalities[0, 1:] = 1

    if len(nullspace) > 0:
        try:
            equalities[1:, 1:] = nullspace
        except ValueError:
            equalities[1:, 1:] = nullspace[:, :, 0]

    pos_constraints = np.concatenate((np.zeros((len(equalities[0])-1, 1)),
                                      np.identity(len(equalities[0]) - 1)),
                                     axis=1)

    return MaterialPolytope(equalities, pos_constraints,
                            return_fractions=return_fractions,
                            independent_endmember_occupancies=endmember_occupancies)


def composite_polytope_at_constrained_composition(composite, composition,
                                                  return_fractions=False):
    """

    """
    c_array = np.empty((composite.n_elements, 1))
    c_array[:, 0] = [-composition[e] if e in composition else 0.
                     for e in composite.elements]

    equalities = np.concatenate((c_array, composite.stoichiometric_array.T),
                                axis=1)

    eoccs = []
    for i, ph in enumerate(composite.phases):
        if isinstance(ph, SolidSolution):
            eoccs.append(ph.solution_model.endmember_occupancies.T)
        else:
            eoccs.append(np.ones((1, 1)))

    eoccs = block_diag(*eoccs)
    inequalities = np.concatenate((np.zeros((len(eoccs), 1)), eoccs),
                                  axis=1)

    return MaterialPolytope(equalities, inequalities,
                            number_type='float',
                            return_fractions=return_fractions)


def feasible_solution_basis_in_component_space(solution, components):
    """
    Note that this function finds the extreme endmembers and finds the
    subset within the components.
    Thus, starting with a solution with a disordered endmember and then
    restricting component range may produce a smaller solution than intended.
    For example, with the endmembers [A] and [A1/2B1/2],
    the extreme endmembers are [A] and [B].
    A component space A--AB will result in only endmember [A] being valid!!
    """

    # 1) Convert components into a matrix
    component_array, component_elements = compositional_array(
        [dictionarize_formula(c) for c in components])

    # 2) Get the full set of endmembers (dependent and otherwise)
    polytope = solution_polytope_from_endmember_occupancies(
        solution.solution_model.endmember_occupancies)
    dependent_sums = polytope.endmembers_as_independent_endmember_proportions

    # 3) Get the endmember compositional array
    independent_endmember_array, endmember_elements = compositional_array(
        solution.endmember_formulae)
    all_endmember_array = dependent_sums.dot(
        independent_endmember_array).round(decimals=12)
    n_all = len(all_endmember_array)

    # 4) Find the endmembers that can be described with
    # a linear combination of components

    # 4a) First, add elements to endmember_elements which are in
    # component_elements
    for el in component_elements:
        if el not in endmember_elements:
            endmember_elements.append(el)
            all_endmember_array = np.concatenate((all_endmember_array,
                                                  np.zeros((n_all, 1))),
                                                 axis=1)

    # 4b) Get rid of endmembers which have elements not in component_elements
    element_indices_for_removal = [i for i, el in enumerate(endmember_elements)
                                   if el not in component_elements]

    mbr_indices_to_rm = []
    for idx in element_indices_for_removal:
        mbr_indices_to_rm.extend(np.nonzero(all_endmember_array[:, idx])[0])
    possible_endmember_indices = np.array([i for i in range(n_all)
                                           if i not in
                                           np.unique(mbr_indices_to_rm)])

    # 4c) Cross-reference indices of elements
    n_el = len(component_elements)
    element_indexing = np.empty(n_el, dtype=int)
    for i in range(n_el):
        element_indexing[i] = endmember_elements.index(component_elements[i])

    # 4d) Find independent endmember set
    def linear_solutions_exist(A, B):
        return [linprog(np.zeros(len(A)), A_eq=A.T, b_eq=b).success
                for b in B]

    B = all_endmember_array[possible_endmember_indices[:, None],
                            element_indexing]
    exist = linear_solutions_exist(component_array, B)
    endmember_indices = possible_endmember_indices[exist]
    independent_indices = endmember_indices[independent_row_indices(
        dependent_sums[endmember_indices])]

    # 5) Return new basis in terms of proportions of the original endmember set
    return dependent_sums[independent_indices]


def decompose_3D_matrix(Wn):
    """
    Decomposes a 3D matrix where E = W_ijk p_i p_j p_k
    into a subregular form where
    E = G_i p_i + WB_ij (1 - p_j + p_i) / 2 + WT_ijk p_i p_j p_k,
    and i < j < k.
    """

    n_mbrs = len(Wn)
    # New endmember components
    # Wn_iii needs to be copied, otherwise just a view onto Wn
    new_endmember_excesses = np.copy(np.einsum('iii->i', Wn))

    # Removal of endmember components from 3D representation
    Wn -= (np.einsum('i, j, k->ijk',
                     new_endmember_excesses, np.ones(n_mbrs),
                     np.ones(n_mbrs))
           + np.einsum('i, j, k->ijk',
                       np.ones(n_mbrs), new_endmember_excesses,
                       np.ones(n_mbrs))
           + np.einsum('i, j, k->ijk',
                       np.ones(n_mbrs), np.ones(n_mbrs),
                       new_endmember_excesses))/3.

    # Transformed 2D components
    # (i=j, i=k, j=k)
    new_binary_matrix = (np.einsum('jki, jk -> ij', Wn, np.identity(n_mbrs))
                         + np.einsum('jik, jk -> ij', Wn, np.identity(n_mbrs))
                         + np.einsum('ijk, jk -> ij', Wn,
                                     np.identity(n_mbrs))).round(decimals=12)

    # Wb is the 3D matrix corresponding to the terms in the binary matrix,
    # such that the two following print statements produce the same answer
    # for a given array of endmember proportions
    Wb = (np.einsum('ijk, ij->ijk', Wn, np.identity(n_mbrs))
          + np.einsum('ijk, jk->ijk', Wn, np.identity(n_mbrs))
          + np.einsum('ijk, ik->ijk', Wn, np.identity(n_mbrs)))

    # Remove binary component from 3D representation
    # The extra terms are needed because the binary term in the formulation
    # of a subregular solution model given by
    # Helffrich and Wood includes ternary components (the sum_k X_k part)..
    Wn -= Wb + (np.einsum('ij, k', new_binary_matrix, np.ones(n_mbrs))
                - np.einsum('ij, ik->ijk', new_binary_matrix,
                            np.identity(n_mbrs))
                - np.einsum('ij, jk->ijk', new_binary_matrix, np.identity(n_mbrs)))/2.

    # Find the 3D components Wijk by adding the elements at
    # the six equivalent positions in the matrix
    new_ternary_terms = []
    for i in range(n_mbrs):
        for j in range(i+1, n_mbrs):
            for k in range(j+1, n_mbrs):
                val = (Wn[i, j, k] + Wn[j, k, i]
                       + Wn[k, i, j] + Wn[k, j, i]
                       + Wn[j, i, k] + Wn[i, k, j]).round(decimals=12)
                if np.abs(val) > 1.e-12:
                    new_ternary_terms.append([i, j, k, val])

    return (new_endmember_excesses, new_binary_matrix, new_ternary_terms)


def _subregular_matrix_conversion(new_basis, binary_matrix,
                                  ternary_terms=None, endmember_excesses=None):
    n_mbrs = len(binary_matrix)
    # Compact 3D representation of original interactions
    W = (np.einsum('i, jk -> ijk', np.ones(n_mbrs), binary_matrix)
         + np.einsum('ij, jk -> ijk', binary_matrix, np.identity(n_mbrs))
         - np.einsum('ij, ik -> ijk', binary_matrix, np.identity(n_mbrs)))/2.

    # Add endmember components to 3D representation
    if endmember_excesses is not None:
        W += (np.einsum('i, j, k->ijk', endmember_excesses,
                        np.ones(n_mbrs), np.ones(n_mbrs))
              + np.einsum('j, i, k->ijk', endmember_excesses,
                          np.ones(n_mbrs), np.ones(n_mbrs))
              + np.einsum('k, i, j->ijk', endmember_excesses,
                          np.ones(n_mbrs), np.ones(n_mbrs)))/3.

    # Add ternary values to 3D representation
    if ternary_terms is not None:
        for i, j, k, val in ternary_terms:
            W[i, j, k] += val

    # Transformation to new 3D representation
    A = new_basis.T
    Wn = np.einsum('il, jm, kn, ijk -> lmn', A, A, A, W)

    new_endmember_excesses, new_binary_terms, new_ternary_terms = decompose_3D_matrix(
        Wn)

    return (new_endmember_excesses, new_binary_terms, new_ternary_terms)


def transform_solution_to_new_basis(solution, new_basis, n_mbrs=None,
                                    solution_name=None, endmember_names=None,
                                    molar_fractions=None):
    """

    """
    new_basis = np.array(new_basis)
    if n_mbrs is None:
        n_mbrs, n_all_mbrs = new_basis.shape
    else:
        _, n_all_mbrs = new_basis.shape

    if solution_name is None:
        name = 'child solution'
    else:
        name = solution_name

    solution_type = solution.solution_type
    if solution_type == 'ideal':
        ESV_modifiers = [[0., 0., 0.] for v in new_basis]

    elif (solution_type == 'asymmetric'
          or solution_type == 'symmetric'):

        A = complete_basis(new_basis).T

        old_alphas = solution.solution_model.alphas
        alphas = np.einsum('i, ij', solution.solution_model.alphas, A)
        inv_diag_alphas = np.diag(1./alphas)
        B = np.einsum('ij, jk, kl->il',
                      np.diag(old_alphas),
                      A, inv_diag_alphas)
        alphas = list(alphas[0:n_mbrs])
        Qe = np.einsum('ik, ij, kl->jl', solution.solution_model.We, B, B)
        Qs = np.einsum('ik, ij, kl->jl', solution.solution_model.Ws, B, B)
        Qv = np.einsum('ik, ij, kl->jl', solution.solution_model.Wv, B, B)

        def new_interactions(Q, n_mbrs):
            return [[float((Q[i, j] + Q[j, i] - Q[i, i] - Q[j, j])
                           * (alphas[i] + alphas[j])/2.)
                     for j in range(i+1, n_mbrs)]
                    for i in range(n_mbrs-1)]

        energy_interaction = new_interactions(Qe, n_mbrs)
        entropy_interaction = new_interactions(Qs, n_mbrs)
        volume_interaction = new_interactions(Qv, n_mbrs)

        ESV_modifiers = [[Qe[i, i] * alphas[i],
                          Qs[i, i] * alphas[i],
                          Qv[i, i] * alphas[i]]
                         for i in range(n_mbrs)]

    elif solution_type == 'subregular':
        full_basis = complete_basis(new_basis)

        def new_interactions(W, n_mbrs):
            return [[[W[i, j], W[j, i]] for j in range(i+1, n_mbrs)]
                    for i in range(n_mbrs-1)]

        # N.B. initial endmember_excesses are zero
        Emod, We, ternary_e = _subregular_matrix_conversion(full_basis,
                                                            solution.solution_model.We,
                                                            solution.solution_model.ternary_terms_e)
        Smod, Ws, ternary_s = _subregular_matrix_conversion(full_basis,
                                                            solution.solution_model.Ws,
                                                            solution.solution_model.ternary_terms_s)
        Vmod, Wv, ternary_v = _subregular_matrix_conversion(full_basis,
                                                            solution.solution_model.Wv,
                                                            solution.solution_model.ternary_terms_v)

        energy_interaction = new_interactions(We, n_mbrs)
        entropy_interaction = new_interactions(Ws, n_mbrs)
        volume_interaction = new_interactions(Wv, n_mbrs)

        ESV_modifiers = [[Emod[i], Smod[i], Vmod[i]] for i in range(n_mbrs)]

    else:
        raise Exception('The function to change basis for the '
                        '{0} solution model has not yet been '
                        'implemented.'.format(solution_type))

    # Create site formulae
    new_occupancies = np.array(new_basis).dot(
        solution.solution_model.endmember_occupancies)
    site_formulae = site_occupancies_to_strings(solution.solution_model.sites,
                                                solution.solution_model.site_multiplicities,
                                                new_occupancies)

    # Create endmembers
    endmembers = []
    for i, vector in enumerate(new_basis):
        nonzero_indices = np.nonzero(vector)[0]
        if len(nonzero_indices) == 1:
            endmembers.append([solution.endmembers[nonzero_indices[0]][0],
                               site_formulae[i]])
        else:
            mbr = CombinedMineral([solution.endmembers[idx][0]
                                   for idx in nonzero_indices],
                                  [vector[idx] for idx in nonzero_indices],
                                  ESV_modifiers[i])
            mbr.params['formula'] = {key: value for (key, value)
                                     in mbr.params['formula'].items()
                                     if value > 1.e-12}
            endmembers.append([mbr, site_formulae[i]])

    if endmember_names is not None:
        for i in range(n_mbrs):
            endmembers[i][0].params['name'] = endmember_names[i]
            endmembers[i][0].name = endmember_names[i]

    if n_mbrs == 1:
        endmembers[0][0].name = name
        endmembers[0][0].parent = solution
        endmembers[0][0].basis = new_basis
        return endmembers[0][0]
    else:
        new_solution = SolidSolution(name=name,
                                     solution_type=solution_type,
                                     endmembers=endmembers,
                                     energy_interaction=energy_interaction,
                                     volume_interaction=volume_interaction,
                                     entropy_interaction=entropy_interaction,
                                     alphas=alphas,
                                     molar_fractions=molar_fractions)
        new_solution.parent = solution
        new_solution.basis = new_basis
        return new_solution


def feasible_solution_in_component_space(solution, components,
                                         solution_name=None,
                                         endmember_names=None,
                                         molar_fractions=None):
    """
    Note that this function finds the extreme endmembers and
    finds the subset within the components.
    Thus, starting with a solution with a disordered endmember and then
    restricting component range
    may produce a smaller solution than intended. For example, with the
    endmembers [A] and [A1/2B1/2], the extreme endmembers are [A] and [B].
    A component space A--AB will result in only endmember [A] being valid!!
    """
    new_basis = feasible_solution_basis_in_component_space(solution,
                                                           components)
    return transform_solution_to_new_basis(solution, new_basis,
                                           solution_name=solution_name,
                                           endmember_names=endmember_names,
                                           molar_fractions=molar_fractions)


def simplify_composite_with_composition(composite, composition):
    """

    """
    polytope = composite_polytope_at_constrained_composition(composite,
                                                             composition,
                                                             return_fractions=True)

    composite_changed = False
    new_phases = []
    mbr_amounts = polytope.endmember_occupancies
    i = 0
    for i_ph, n_mbrs in enumerate(composite.endmembers_per_phase):
        ph = composite.phases[i_ph]

        amounts = mbr_amounts[:, i:i+n_mbrs].astype(float)
        i += n_mbrs

        rank = np.linalg.matrix_rank(amounts, tol=1.e-8)

        if rank < n_mbrs:

            if isinstance(ph, SolidSolution) and rank > 0:

                if len(amounts) > 1:
                    c_mean = np.mean(amounts, axis=0)
                else:
                    c_mean = amounts[0]

                poly = solution_polytope_from_endmember_occupancies(
                    ph.solution_model.endmember_occupancies)
                dmbrs = poly.endmembers_as_independent_endmember_proportions

                x = cp.Variable(dmbrs.shape[0])
                objective = cp.Minimize(cp.sum_squares(x))
                constraints = [dmbrs.T@x == c_mean, x >= 0]

                prob = cp.Problem(objective, constraints)
                prob.solve()

                mbr_indices = np.argsort(x.value)[::-1]
                ind_indices = [i for i in mbr_indices
                               if x.value[i] > 1.e-6]
                new_basis = dmbrs[ind_indices]

                if len(new_basis) < ph.n_endmembers:
                    logging.info(f'Phase {i_ph} ({ph.name}) is '
                                 'rank-deficient ({rank} < {n_mbrs}). '
                                 'The transformed solution is described '
                                 f'using {len(new_basis)} endmembers.')

                    composite_changed = True
                    soln = transform_solution_to_new_basis(ph, new_basis)
                    new_phases.append(soln)
                else:
                    logging.info('This solution is rank-deficient '
                                 f'({rank} < {n_mbrs}), '
                                 'but its composition requires all '
                                 'independent endmembers.')
            else:
                composite_changed = True
                logging.info(f'Phase {i_ph} ({ph.name}) removed from '
                             'composite (rank = 0).')
        else:
            new_phases.append(ph)

    if composite_changed:
        return Composite(new_phases)
    else:
        return composite
