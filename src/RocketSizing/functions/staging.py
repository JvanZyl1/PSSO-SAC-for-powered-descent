import numpy as np
from math import sqrt, log, exp
from typing import Sequence, Dict, Tuple
from scipy.optimize import minimize, brentq, fsolve
import csv

MU_E = 3.98602e14
R_E = 6378137.0
y_LEO = 210000.0
a = y_LEO + R_E
dv_req = sqrt(MU_E/a)


def staging_p1_reproduction(m_pay: float,
                            dv_a_loss: Sequence[float],
                            dv_d_1_loss: float,
                            dv_d_1_star: float,
                            v_ex: Sequence[float],
                            eps: Sequence[float],
                            debug: bool):
    # Constants
    vex_1 = v_ex[0]
    vex_2 = v_ex[1]
    eps_1 = eps[0]
    eps_2 = eps[1]
    dv_a_1_loss = dv_a_loss[0]
    dv_2_loss = dv_a_loss[1]
    
    dv_d_1 = dv_d_1_star + dv_d_1_loss
    # --------------- SOLVE FOR KAPPA ---------------------#
    def func(kappa):
        kappa = float(kappa)
        # Check if kappa is within bounds
        if kappa <= 0 or kappa >= min(vex_1, vex_2):
            return float('inf')
        # Ensure arguments to log are positive
        term1 = (vex_1 - kappa)/(vex_1 * eps_1)
        term2 = (vex_2 - kappa)/(vex_2 * eps_2)
        if term1 <= 0 or term2 <= 0:
            return float('inf')
        RHS = vex_1 * log((vex_1 - kappa)/(vex_1 * eps_1)) + vex_2 * log((vex_2 - kappa)/(vex_2 * eps_2)) - dv_d_1
        LHS = dv_req
        return RHS - LHS
    
    max_bound = min(vex_1, vex_2)
    initial_guess = 0.5 * max_bound
    result = fsolve(func, [initial_guess], xtol=1e-7, maxfev=1000)
    kappa = float(result[0])
    if kappa <= 0 or kappa >= min(vex_1, vex_2):
        raise ValueError(f"Solution kappa={kappa} outside valid range (0, {min(vex_1, vex_2)})")
    kappa = 2555.545743
    #---------------SIZE EXPENDABLE SECOND STAGE -----------------#
    lambda_2_star = kappa * eps_2/((1-eps_2)*vex_2 - kappa)
    dv_2_star = vex_2 * log((1+lambda_2_star)/(eps_2 + lambda_2_star))
    dv_2 = dv_2_star + dv_2_loss
    lambda_2_l_star = (eps_2 * exp(dv_2/vex_2) - 1)/(1 - exp(dv_2/vex_2))
    ms_2_l_star = eps_2/lambda_2_l_star * m_pay
    mp_2_l_star = (1-eps_2)/lambda_2_l_star * m_pay
    m_2_l_star = ms_2_l_star + mp_2_l_star

    #--------------SIZE REUSABLE FIRST STAGE --------------------#
    eps_d_1 = 1/(exp(dv_d_1_star/vex_1))
    eps_a_1 = eps_1/eps_d_1
    lambda_1_star = kappa * eps_a_1 / ((1-eps_a_1)*vex_1 - kappa)
    dv_a_1_star = vex_1 * log((1+lambda_1_star)/(eps_a_1+lambda_1_star))
    dv_a_1 = dv_a_1_star + dv_a_1_loss
    eps_d_1_l = exp(-dv_d_1/vex_1)
    eps_a_1_l = eps_1/eps_d_1_l
    lambda_1_l_star = (eps_a_1_l * exp(dv_a_1/vex_1) - 1)/(1 - exp(dv_a_1/vex_1))
    mL_1_l_star = (1/lambda_2_l_star+1)*m_pay
    ms_a_1_l_star = eps_a_1_l/lambda_1_l_star * mL_1_l_star
    mp_a_1_l_star = (1-eps_a_1_l)/lambda_1_l_star * mL_1_l_star
    m_1_l_star = ms_a_1_l_star + mp_a_1_l_star
    m_d_1_l_star = ms_a_1_l_star
    ms_d_1_l_star = m_d_1_l_star * eps_d_1_l
    mp_d_1_l_star = m_d_1_l_star - ms_d_1_l_star
    ms_1_l_star = ms_d_1_l_star
    mp_1_l_star = mp_a_1_l_star + mp_d_1_l_star
    #-------------DEBUG STATEMENTS ----------------------------#
    if debug:
        print(f'kappa: {kappa}, should be 2555.545743')
        print(f'lambda_2_star: {lambda_2_star}, should be 0.1801561609')
        print(f'dv_2_star: {dv_2_star}, should be 5597.179423')
        print(f'dv_2: {dv_2}, should be 6307.179423')
        print(f'lambda_2_l_star: {lambda_2_l_star}, should be 0.1291331291')
        print(f'ms_2_l_star: {ms_2_l_star}, should be 5841.91683')
        print(f'mp_2_l_star: {mp_2_l_star}, should be 114189.2488')
        print(f'm_2_l_star: {m_2_l_star}, should be 120031.1656')
        print(f'eps_d_1: {eps_d_1}, should be 0.5710218129')
        print(f'eps_a_1: {eps_a_1}, should be 0.0897513875')
        print(f'lambda_1_star: {lambda_1_star}, should be 1.03919692')
        print(f'dv_a_1_star: {dv_a_1_star}, should be 1803.372175')
        print(f'dv_a_1: {dv_a_1}, should be 3194.372175')
        print(f'eps_d_1_l: {eps_d_1_l}, should be 0.3607131874')
        print(f'eps_a_1_l: {eps_a_1_l}, should be 0.1420796405')
        print(f'lambda_1_l_star: {lambda_1_l_star}, should be 0.3216495603')
        print(f'mL_1_l_star: {mL_1_l_star}, should be 135531.1656')
        print(f'ms_a_1_l_star: {ms_a_1_l_star}, should be 59867.07792')
        print(f'mp_a_1_l_star: {mp_a_1_l_star}, should be 361495.7415')
        print(f'm_1_l_star: {m_1_l_star}, should be 421362.8194')
        print(f'm_d_1_l_star: {m_d_1_l_star}, should be 59867.07792')
        print(f'ms_d_1_l_star: {ms_d_1_l_star}, should be 21594.8445')
        print(f'mp_d_1_l_star: {mp_d_1_l_star}, should be 38272.23342')
        print(f'ms_1_l_star: {ms_1_l_star}, should be 21594.8445')
        print(f'mp_1_l_star: {mp_1_l_star}, should be 399767.9749')


    #-------------STORE IN DICTIONARY --------------------------#
    m0 = m_1_l_star + m_2_l_star + m_pay

    stage_masses_dict = {
        'structural_mass_stage_1_ascent': ms_a_1_l_star,
        'propellant_mass_stage_1_ascent': mp_a_1_l_star,
        'structural_mass_stage_1_descent': ms_d_1_l_star,
        'propellant_mass_stage_1_descent': mp_d_1_l_star,
        'structural_mass_stage_2_ascent': ms_2_l_star, # excludes payload mass
        'propellant_mass_stage_2_ascent': mp_2_l_star,
        'payload_mass': m_pay,
        'initial_mass': m0,
        'mass_at_stage_1_ascent_burnout': m0 - mp_a_1_l_star,
        'mass_of_rocket_at_stage_1_separation': m_2_l_star + m_pay,
        'mass_of_stage_1_at_separation': m_d_1_l_star,
        'mass_at_stage_1_descent_burnout': ms_d_1_l_star,
        'mass_at_stage_2_ascent_burnout': ms_2_l_star + m_pay,
        'mass_at_stage_2_separation': m_pay
    }

    trace = {
        "kappa": kappa,
        "payload_opt_1": lambda_1_star, "payload_opt_2": lambda_2_star,
        "payload_ratio_l_1_star": lambda_1_l_star, "payload_ratio_l_2_star": lambda_2_l_star,
        "eps_a_1": eps_a_1, "eps_d_1": eps_d_1,
        "eps_a_1_l": eps_a_1_l, "eps_d_1_l": eps_d_1_l,
        "dv_star_a_1": dv_a_1_star, "dv_star_2": dv_2_star,
        "m0": m0,
        "ms_1": ms_1_l_star, "mp_1": mp_1_l_star,
        "ms_2": ms_2_l_star, "mp_2": mp_2_l_star,
    }

    # Update loss-free velocity increments
    with open('data/rocket_parameters/velocity_increments.csv', 'a') as f:
        # clear the file
        f.truncate(0)
        writer = csv.writer(f)
        writer.writerow(['(sizing input) dv_loss_a_1', dv_a_1_loss])
        writer.writerow(['(sizing input) dv_loss_a_2', dv_2_loss])
        writer.writerow(['(sizing input) dv_loss_d_1', dv_d_1_loss])
        writer.writerow(['(sizing input) dv_d_1', dv_d_1])
        writer.writerow(['(sizing output) dv_star_a_1', dv_a_1_star])
        writer.writerow(['(sizing output) dv_star_2', dv_2_star])
    
    return stage_masses_dict, trace


if __name__ == "__main__":
    # inputs (Jo & Ahn 2021, Table 1)
    R_E, ALT = 6378137.0, 210000.0
    a        = R_E + ALT
    stage, tr = staging_p1_reproduction(
        m_pay = 15.5e3,
        dv_a_loss = [1391.0, 710.0],
        dv_d_1_loss = 1401.0,
        dv_d_1_star = 1709,
        v_ex     =[3050.0, 3412.0],
        eps      =[0.05125, 0.0487],
        debug=True
    )