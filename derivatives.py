# TaylorSWIF derivatives
from pagos.gas import ice, abn, calc_Ceq, calc_dCeq_dT
from pagos.builtin_models import taylor_swif, taylor_swift, dwarf, qs_dwarf

def TaylorSWIF_dCdR(gas, T_r, S, p, A):
    Ceq = calc_Ceq(gas, T_r, S, p)
    return (Ceq + A * abn(gas)) * (1 - ice(gas))

def TaylorSWIFT_dCdR(gas, T_r, S, p, A):
    Ceq = calc_Ceq(gas, T_r, S, p)
    return (Ceq + A * abn(gas)) * (1 - ice(gas)**2)

def DWARF_dCdomega(gas, T_r, S, p, omega, zeta):
    Ceq = calc_Ceq(gas, T_r, S, p)
    return (Ceq + zeta * abn(gas)) * (1 - ice(gas)) / ((ice(gas) - 1)*omega + 1)**2

def QSDWARF_dCdomega(gas, T, S, p, omega, zeta, T_r):
    Ceq = calc_Ceq(gas, T_r, S, p)
    CeqT = calc_Ceq(gas, T, S, p)
    dCeq_dT = calc_dCeq_dT(gas, T, S, p)
    return (Ceq + zeta * abn(gas)) * (1 - ice(gas)) / ((ice(gas) - 1)*omega + 1 + dCeq_dT / CeqT * (T - T_r))**2


def TaylorSWIF_dCdA(gas, R):
    return abn(gas) * (1 - R*(ice(gas) - 1))

def TaylorSWIFT_dCdA(gas, R):
    return abn(gas) * (1 - R*(ice(gas)**2 - 1))

def DWARF_dCdzeta(gas, omega):
    return abn(gas) / (1 + omega*(ice(gas) - 1))

def QSDWARF_dCdzeta(gas, T, S, p, omega, zeta, T_r):
    C_QS = qs_dwarf(gas, T, S, p, omega, zeta, T_r)
    Ceq = calc_Ceq(gas, T_r, S, p)
    return C_QS * abn(gas) / (Ceq + zeta * abn(gas))


def TaylorSWIF_dCdCeq(gas, R):
    return 1 - R * (ice(gas) - 1)

def TaylorSWIFT_dCdCeq(gas, R):
    return 1 - R * (ice(gas)**2 - 1)

def DWARF_dCdCeq(gas, omega):
    return 1 / (1 + omega * (ice(gas) - 1))

def QSDWARF_dCdCeq(gas, T, S, p, omega, zeta, T_r):
    C_QS = qs_dwarf(gas, T, S, p, omega, zeta, T_r)
    Ceq = calc_Ceq(gas, T_r, S, p)
    return C_QS / (Ceq + zeta * abn(gas))


def QSDWARF_dCdT_r(gas, T, S, p, omega, zeta, T_r):
    dCeq_dT = calc_dCeq_dT(gas, T, S, p)
    Ceq = calc_Ceq(gas, T_r, S, p)
    CeqT = calc_Ceq(gas, T, S, p)
    return dCeq_dT * (Ceq + zeta*abn(gas)) / (CeqT * (dCeq_dT/CeqT * (T - T_r) + (ice(gas) - 1)*omega + 1)**2)


def calc_sigma_TaylorSWIF(gas, T_r, S, p, R, A, sigma_T_r, sigma_R, sigma_A):
    dCdR = TaylorSWIF_dCdR(gas, T_r, S, p, A)
    dCdCeq = TaylorSWIF_dCdCeq(gas, R)
    dCdA = TaylorSWIF_dCdA(gas, R)
    dCeqdT = calc_dCeq_dT(gas, T_r, S, p)
    """print("dCdR TaylorSWIF:", dCdR)
    print("dCdCeq TaylorSWIF:", dCdCeq)
    print("dCdA TaylorSWIF:", dCdA)
    print("dCeqdT TaylorSWIF:", dCeqdT)
    print("sigma_T_r:", sigma_T_r)
    print("sigma_R:", sigma_R)
    print("sigma_A:", sigma_A)"""

    var = (dCdR * sigma_R)**2 + (dCdCeq * dCeqdT * sigma_T_r)**2 + (dCdA * sigma_A)**2
    #print("sigma_TaylorSWIF:", var**0.5)
    return var**0.5

def calc_sigma_TaylorSWIFT(gas, T_r, S, p, R, A, sigma_T_r, sigma_R, sigma_A):
    dCdR = TaylorSWIFT_dCdR(gas, T_r, S, p, A)
    dCdCeq = TaylorSWIFT_dCdCeq(gas, R)
    dCdA = TaylorSWIFT_dCdA(gas, R)
    dCeqdT = calc_dCeq_dT(gas, T_r, S, p)
    """print("dCdR TaylorSWIFT:", dCdR)
    print("dCdCeq TaylorSWIFT:", dCdCeq)
    print("dCdA TaylorSWIFT:", dCdA)
    print("dCeqdT TaylorSWIFT:", dCeqdT)
    print("sigma_T_r:", sigma_T_r)
    print("sigma_R:", sigma_R)
    print("sigma_A:", sigma_A)"""

    var = (dCdR * sigma_R)**2 + (dCdCeq * dCeqdT * sigma_T_r)**2 + (dCdA * sigma_A)**2
    #print("sigma_TaylorSWIFT:", var**0.5)
    return var**0.5

def calc_sigma_DWARF(gas, T_r, S, p, omega, zeta, sigma_T_r, sigma_omega, sigma_zeta):
    dCdomega = DWARF_dCdomega(gas, T_r, S, p, omega, zeta)
    dCdCeq = DWARF_dCdCeq(gas, omega)
    dCdzeta = DWARF_dCdzeta(gas, omega)
    dCeqdT = calc_dCeq_dT(gas, T_r, S, p)
    """print("dCdomega DWARF:", dCdomega)
    print("dCdCeq DWARF:", dCdCeq)
    print("dCdzeta DWARF:", dCdzeta)
    print("dCeqdT DWARF:", dCeqdT)
    print("sigma_T_r:", sigma_T_r)
    print("sigma_omega:", sigma_omega)
    print("sigma_zeta:", sigma_zeta)"""

    var = (dCdomega * sigma_omega)**2 + (dCdCeq * dCeqdT * sigma_T_r)**2 + (dCdzeta * sigma_zeta)**2
    #print("sigma_DWARF:", var**0.5)
    return var**0.5

def calc_sigma_QSDWARF(gas, T, S, p, omega, zeta, T_r, sigma_omega, sigma_zeta, sigma_T_r):
    dCdomega = QSDWARF_dCdomega(gas, T, S, p, omega, zeta, T_r)
    dCdT_r = QSDWARF_dCdT_r(gas, T, S, p, omega, zeta, T_r)
    dCdCeq = QSDWARF_dCdCeq(gas, T, S, p, omega, zeta, T_r)
    dCdzeta = QSDWARF_dCdzeta(gas, T, S, p, omega, zeta, T_r)
    dCeqdT = calc_dCeq_dT(gas, T_r, S, p)
    """print("dCdomega QSDWARF:", dCdomega)
    print("dCdT_r QSDWARF:", dCdT_r)
    print("dCdCeq QSDWARF:", dCdCeq)
    print("dCdzeta QSDWARF:", dCdzeta)
    print("dCeqdT QSDWARF:", dCeqdT)
    print("sigma_T_r:", sigma_T_r)
    print("sigma_omega:", sigma_omega)
    print("sigma_zeta:", sigma_zeta)"""

    var = (dCdomega * sigma_omega)**2 + ((dCdT_r + dCdCeq * dCeqdT) * sigma_T_r)**2 + (dCdzeta * sigma_zeta)**2
    #print("sigma_QSDWARF:", var**0.5)
    return var**0.5