#%%
import pagos
from pagos import GasExchangeModel, calc_Ceq
from pagos.builtin_models import taylor_swif, taylor_swift, dwarf, qs_dwarf
from pagos.constants import NOBLEGASES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pagos import Q

plt.rcParams['text.usetex'] = True
# %% data import
vacao_data_raw = pd.read_csv('data/NG_data_all_avg_final_adjusted.CSV')

vacao_data = vacao_data_raw.copy()
vacao_data['Minus Depth'] = -1 * vacao_data['Depth']

vacao_colors = {5:'#D81B60', 8:'#1E88E5', 16:'#FFC107', 20:'#91FF5F', 28:'#C7C7C7', 46:'#000000'}
plot_colors = [vacao_colors[s] for s in vacao_data['Station']]
gas_factors = {'He':1e8, 'Ne':1e7, 'Ar':1e4, 'Kr':1e7, 'Xe':1e8}
# %% fitting
ModelTaylorSWIF = GasExchangeModel(taylor_swif, ('degC', 'permille', 'atm', '', 'cc/g'), 'cc/g')
ModelTaylorSWIFT = GasExchangeModel(taylor_swift, ('degC', 'permille', 'atm', '', 'cc/g'), 'cc/g')
ModelDWARF = GasExchangeModel(dwarf, ('degC', 'permille', 'atm', '', 'cc/g'), 'cc/g')
ModelQSDWARF = GasExchangeModel(qs_dwarf, ('degC', 'permille', 'atm', '', 'cc/g', 'degC'), 'cc/g')

# do fits
FitTaylorSWIF = ModelTaylorSWIF.fit(vacao_data, ['T_r', 'R', 'A'], [0, 0.01, 1e-5], NOBLEGASES, tqdm_bar=True)
FitTaylorSWIFT = ModelTaylorSWIFT.fit(vacao_data, ['T_r', 'R', 'A'], [0, 0.01, 1e-5], NOBLEGASES, tqdm_bar=True)
FitDWARF = ModelDWARF.fit(vacao_data, ['T_r', 'omega', 'zeta'], [0, 0.01, 1e-5], NOBLEGASES, tqdm_bar=True)
FitQSDWARF = ModelQSDWARF.fit(vacao_data, ['T_r', 'omega', 'zeta'], [0, 0.01, 1e-5], NOBLEGASES, tqdm_bar=True)

FitTaylorSWIF_C = ModelTaylorSWIF.fit(vacao_data, ['T_r', 'R', 'A'], [0, 0.01, 1e-5], ['He', 'Ar', 'Kr', 'Xe'], tqdm_bar=True)#, constraints=[(-1.9, np.infty), (0, np.infty), (-np.infty, np.infty)])
FitTaylorSWIFT_C = ModelTaylorSWIFT.fit(vacao_data, ['T_r', 'R', 'A'], [0, 0.01, 1e-5], ['He', 'Ar', 'Kr', 'Xe'], tqdm_bar=True)#, constraints=[(-1.9, np.infty), (0, np.infty), (-np.infty, np.infty)])
FitDWARF_C = ModelDWARF.fit(vacao_data, ['T_r', 'omega', 'zeta'], [0, 0.01, 1e-5], ['He', 'Ar', 'Kr', 'Xe'], tqdm_bar=True)#, constraints=[(-1.9, np.infty), (0, np.infty), (-np.infty, np.infty)])
FitQSDWARF_C = ModelQSDWARF.fit(vacao_data, ['T_r', 'omega', 'zeta'], [0, 0.01, 1e-5], ['He', 'Ar', 'Kr', 'Xe'], tqdm_bar=True)#, constraints=[(-1.9, np.infty), (0, np.infty), (-np.infty, np.infty)])

# append depth to dataframes
FitTaylorSWIF = pd.concat((FitTaylorSWIF, vacao_data[['Depth', 'Minus Depth', 'Station']]), axis=1)
FitTaylorSWIFT = pd.concat((FitTaylorSWIFT, vacao_data[['Depth', 'Minus Depth', 'Station']]), axis=1)
FitDWARF = pd.concat((FitDWARF, vacao_data[['Depth', 'Minus Depth', 'Station']]), axis=1)
FitQSDWARF = pd.concat((FitQSDWARF, vacao_data[['Depth', 'Minus Depth', 'Station']]), axis=1)

FitTaylorSWIF_C = pd.concat((FitTaylorSWIF_C, vacao_data[['Depth', 'Minus Depth', 'Station']]), axis=1)
FitTaylorSWIFT_C = pd.concat((FitTaylorSWIFT_C, vacao_data[['Depth', 'Minus Depth', 'Station']]), axis=1)
FitDWARF_C = pd.concat((FitDWARF_C, vacao_data[['Depth', 'Minus Depth', 'Station']]), axis=1)
FitQSDWARF_C = pd.concat((FitQSDWARF_C, vacao_data[['Depth', 'Minus Depth', 'Station']]), axis=1)

# %% CFC12 and SF6 saturations
import derivatives 

# import MC data
vacao_data_MC_TaylorSWIF = pd.read_csv('data/Comparison_Results_TAYLORSWIF_adjusted.csv')
vacao_data_MC_TaylorSWIFT = pd.read_csv('data/Comparison_Results_TAYLORSWIFT_adjusted.csv')
vacao_data_MC_DWARF = pd.read_csv('data/Comparison_Results_DWARF_adjusted.csv')
vacao_data_MC_QSDWARF = pd.read_csv('data/Comparison_Results_QSDWARF_adjusted.csv')


# combine with measured T and S data
append_series = append_series = pd.DataFrame({'T':[vacao_data['T'].iloc[i] for i in vacao_data_MC_TaylorSWIF['Sample_idx']]})

vacao_data_MC_TaylorSWIF = pd.concat((vacao_data_MC_TaylorSWIF, append_series), axis=1)
vacao_data_MC_TaylorSWIFT = pd.concat((vacao_data_MC_TaylorSWIFT, append_series), axis=1)
vacao_data_MC_DWARF = pd.concat((vacao_data_MC_DWARF, append_series), axis=1)
vacao_data_MC_QSDWARF = pd.concat((vacao_data_MC_QSDWARF, append_series), axis=1)

#%% Set ice fractionation coefficients
#pagos.constants.ICE_FRACTIONATION_COEFFS['CFC12'] = 0.5
#pagos.constants.ICE_FRACTIONATION_COEFFS['SF6'] = 0.5

#%%
# TaylorSWIF
dict_to_save = {'CFC12 sat': [], 'CFC12 sat err': [], 'SF6 sat': [], 'SF6 sat err': []}
for i in vacao_data_MC_TaylorSWIF['Sample_idx'].unique():
    R =  vacao_data_MC_TaylorSWIF['MC R'][vacao_data_MC_TaylorSWIF['Sample_idx']==i].iloc[0]
    A =  vacao_data_MC_TaylorSWIF['MC A'][vacao_data_MC_TaylorSWIF['Sample_idx']==i].iloc[0]
    T_r =  vacao_data_MC_TaylorSWIF['MC T_r'][vacao_data_MC_TaylorSWIF['Sample_idx']==i].iloc[0]
    sigma_R = vacao_data_MC_TaylorSWIF['MC R std'][vacao_data_MC_TaylorSWIF['Sample_idx']==i].iloc[0]
    sigma_A = vacao_data_MC_TaylorSWIF['MC A std'][vacao_data_MC_TaylorSWIF['Sample_idx']==i].iloc[0]
    sigma_T_r = vacao_data_MC_TaylorSWIF['MC T_r std'][vacao_data_MC_TaylorSWIF['Sample_idx']==i].iloc[0]
    S = vacao_data_MC_TaylorSWIF['S'][vacao_data_MC_TaylorSWIF['Sample_idx']==i].iloc[0]
    T = vacao_data_MC_TaylorSWIF['T'][vacao_data_MC_TaylorSWIF['Sample_idx']==i].iloc[0]
    
    # calculate saturations
    calc_cfc12 = ModelTaylorSWIF.run('CFC12', T_r, S, 1, R, A).magnitude
    eq_cfc12 = calc_Ceq('CFC12', T, S, 1)
    delta_cfc12 = calc_cfc12 / eq_cfc12 - 1

    calc_sf6 = ModelTaylorSWIF.run('SF6', T_r, S, 1, R, A).magnitude
    eq_sf6 = calc_Ceq('SF6', T, S, 1)
    delta_sf6 = calc_sf6 / eq_sf6 - 1

    dict_to_save['CFC12 sat'].append(delta_cfc12)
    dict_to_save['SF6 sat'].append(delta_sf6)

    # calculate errors'
    sigma_mod_cfc12 = derivatives.calc_sigma_TaylorSWIF('CFC12', T_r, S, 1, R, A, sigma_T_r, sigma_R, sigma_A)
    sigma_mod_sf6 = derivatives.calc_sigma_TaylorSWIF('SF6', T_r, S, 1, R, A, sigma_T_r, sigma_R, sigma_A)
    sigma_cfc12 = sigma_mod_cfc12 / eq_cfc12
    sigma_sf6 = sigma_mod_sf6 / eq_sf6

    dict_to_save['CFC12 sat err'].append(sigma_cfc12)
    dict_to_save['SF6 sat err'].append(sigma_sf6)

TaylorSWIF_TTs = pd.DataFrame(dict_to_save)
TaylorSWIF_TTs['Station'] = vacao_data_MC_TaylorSWIF['Station'].values
TaylorSWIF_TTs['Depth'] = vacao_data_MC_TaylorSWIF['Depth'].values
TaylorSWIF_TTs.to_csv('data/TaylorSWIF_TTs.csv', index=False)
#%%

# TaylorSWIFT
dict_to_save = {'CFC12 sat': [], 'CFC12 sat err': [], 'SF6 sat': [], 'SF6 sat err': []}
for i in vacao_data_MC_TaylorSWIFT['Sample_idx'].unique():
    R = vacao_data_MC_TaylorSWIFT['MC R'][vacao_data_MC_TaylorSWIFT['Sample_idx']==i].iloc[0]
    A = vacao_data_MC_TaylorSWIFT['MC A'][vacao_data_MC_TaylorSWIFT['Sample_idx']==i].iloc[0]
    T_r = vacao_data_MC_TaylorSWIFT['MC T_r'][vacao_data_MC_TaylorSWIFT['Sample_idx']==i].iloc[0]
    sigma_R = vacao_data_MC_TaylorSWIFT['MC R std'][vacao_data_MC_TaylorSWIFT['Sample_idx']==i].iloc[0]
    sigma_A = vacao_data_MC_TaylorSWIFT['MC A std'][vacao_data_MC_TaylorSWIFT['Sample_idx']==i].iloc[0]
    sigma_T_r = vacao_data_MC_TaylorSWIFT['MC T_r std'][vacao_data_MC_TaylorSWIFT['Sample_idx']==i].iloc[0]
    S = vacao_data_MC_TaylorSWIFT['S'][vacao_data_MC_TaylorSWIFT['Sample_idx']==i].iloc[0]
    T = vacao_data_MC_TaylorSWIFT['T'][vacao_data_MC_TaylorSWIFT['Sample_idx']==i].iloc[0]
    
    # calculate saturations
    calc_cfc12 = ModelTaylorSWIFT.run('CFC12', T_r, S, 1, R, A).magnitude
    eq_cfc12 = calc_Ceq('CFC12', T, S, 1)
    delta_cfc12 = calc_cfc12 / eq_cfc12 - 1

    calc_sf6 = ModelTaylorSWIFT.run('SF6', T_r, S, 1, R, A).magnitude
    eq_sf6 = calc_Ceq('SF6', T, S, 1)
    delta_sf6 = calc_sf6 / eq_sf6 - 1

    dict_to_save['CFC12 sat'].append(delta_cfc12)
    dict_to_save['SF6 sat'].append(delta_sf6)

    # calculate errors'
    sigma_mod_cfc12 = derivatives.calc_sigma_TaylorSWIFT('CFC12', T_r, S, 1, R, A, sigma_T_r, sigma_R, sigma_A)
    sigma_mod_sf6 = derivatives.calc_sigma_TaylorSWIFT('SF6', T_r, S, 1, R, A, sigma_T_r, sigma_R, sigma_A)
    sigma_cfc12 = sigma_mod_cfc12 / eq_cfc12
    sigma_sf6 = sigma_mod_sf6 / eq_sf6

    dict_to_save['CFC12 sat err'].append(sigma_cfc12)
    dict_to_save['SF6 sat err'].append(sigma_sf6)

TaylorSWIFT_TTs = pd.DataFrame(dict_to_save)
TaylorSWIFT_TTs['Station'] = vacao_data_MC_TaylorSWIFT['Station'].values
TaylorSWIFT_TTs['Depth'] = vacao_data_MC_TaylorSWIFT['Depth'].values
TaylorSWIFT_TTs.to_csv('data/TaylorSWIFT_TTs.csv', index=False)

#%%
# DWARF
dict_to_save = {'CFC12 sat': [], 'CFC12 sat err': [], 'SF6 sat': [], 'SF6 sat err': []}
for i in vacao_data_MC_DWARF['Sample_idx'].unique():
    omega = vacao_data_MC_DWARF['MC omega'][vacao_data_MC_DWARF['Sample_idx']==i].iloc[0]
    zeta = vacao_data_MC_DWARF['MC zeta'][vacao_data_MC_DWARF['Sample_idx']==i].iloc[0]
    T_r = vacao_data_MC_DWARF['MC T_r'][vacao_data_MC_DWARF['Sample_idx']==i].iloc[0]
    sigma_omega = vacao_data_MC_DWARF['MC omega std'][vacao_data_MC_DWARF['Sample_idx']==i].iloc[0]
    sigma_zeta = vacao_data_MC_DWARF['MC zeta std'][vacao_data_MC_DWARF['Sample_idx']==i].iloc[0]
    sigma_T_r = vacao_data_MC_DWARF['MC T_r std'][vacao_data_MC_DWARF['Sample_idx']==i].iloc[0]
    S = vacao_data_MC_DWARF['S'][vacao_data_MC_DWARF['Sample_idx']==i].iloc[0]
    T = vacao_data_MC_DWARF['T'][vacao_data_MC_DWARF['Sample_idx']==i].iloc[0]

    # calculate saturations
    calc_cfc12 = ModelDWARF.run('CFC12', T_r, S, 1, omega, zeta).magnitude
    eq_cfc12 = calc_Ceq('CFC12', T, S, 1)
    delta_cfc12 = calc_cfc12 / eq_cfc12 - 1

    calc_sf6 = ModelDWARF.run('SF6', T_r, S, 1, omega, zeta).magnitude
    eq_sf6 = calc_Ceq('SF6', T, S, 1)
    delta_sf6 = calc_sf6 / eq_sf6 - 1

    dict_to_save['CFC12 sat'].append(delta_cfc12)
    dict_to_save['SF6 sat'].append(delta_sf6)

    # calculate errors'
    sigma_mod_cfc12 = derivatives.calc_sigma_DWARF('CFC12', T_r, S, 1, omega, zeta, sigma_T_r, sigma_omega, sigma_zeta)
    sigma_mod_sf6 = derivatives.calc_sigma_DWARF('SF6', T_r, S, 1, omega, zeta, sigma_T_r, sigma_omega, sigma_zeta)
    sigma_cfc12 = sigma_mod_cfc12 / eq_cfc12
    sigma_sf6 = sigma_mod_sf6 / eq_sf6

    dict_to_save['CFC12 sat err'].append(sigma_cfc12)
    dict_to_save['SF6 sat err'].append(sigma_sf6)

DWARF_TTs = pd.DataFrame(dict_to_save)
DWARF_TTs['Station'] = vacao_data_MC_DWARF['Station'].values
DWARF_TTs['Depth'] = vacao_data_MC_DWARF['Depth'].values
DWARF_TTs.to_csv('data/DWARF_TTs.csv', index=False)

#%%
# QSDWARF
dict_to_save = {'CFC12 sat': [], 'CFC12 sat err': [], 'SF6 sat': [], 'SF6 sat err': []}
for i in vacao_data_MC_QSDWARF['Sample_idx'].unique():
    omega = vacao_data_MC_QSDWARF['MC omega'][vacao_data_MC_QSDWARF['Sample_idx']==i].iloc[0]
    zeta = vacao_data_MC_QSDWARF['MC zeta'][vacao_data_MC_QSDWARF['Sample_idx']==i].iloc[0]
    T_r = vacao_data_MC_QSDWARF['MC T_r'][vacao_data_MC_QSDWARF['Sample_idx']==i].iloc[0]
    sigma_omega = vacao_data_MC_QSDWARF['MC omega std'][vacao_data_MC_QSDWARF['Sample_idx']==i].iloc[0]
    sigma_zeta = vacao_data_MC_QSDWARF['MC zeta std'][vacao_data_MC_QSDWARF['Sample_idx']==i].iloc[0]
    sigma_T_r = vacao_data_MC_QSDWARF['MC T_r std'][vacao_data_MC_QSDWARF['Sample_idx']==i].iloc[0]
    S = vacao_data_MC_QSDWARF['S'][vacao_data_MC_QSDWARF['Sample_idx']==i].iloc[0]
    T = vacao_data_MC_QSDWARF['T'][vacao_data_MC_QSDWARF['Sample_idx']==i].iloc[0]
    
    # calculate saturations
    calc_cfc12 = ModelQSDWARF.run('CFC12', T, S, 1, omega, zeta, T_r).magnitude
    eq_cfc12 = calc_Ceq('CFC12', T, S, 1)
    delta_cfc12 = calc_cfc12 / eq_cfc12 - 1

    calc_sf6 = ModelQSDWARF.run('SF6', T, S, 1, omega, zeta, T_r).magnitude
    eq_sf6 = calc_Ceq('SF6', T, S, 1)
    delta_sf6 = calc_sf6 / eq_sf6 - 1

    dict_to_save['CFC12 sat'].append(delta_cfc12)
    dict_to_save['SF6 sat'].append(delta_sf6)

    # calculate errors'
    sigma_mod_cfc12 = derivatives.calc_sigma_QSDWARF('CFC12', T, S, 1, omega, zeta, T_r, sigma_omega, sigma_zeta, sigma_T_r)
    sigma_mod_sf6 = derivatives.calc_sigma_QSDWARF('SF6', T, S, 1, omega, zeta, T_r, sigma_omega, sigma_zeta, sigma_T_r)
    sigma_cfc12 = sigma_mod_cfc12 / eq_cfc12
    sigma_sf6 = sigma_mod_sf6 / eq_sf6

    dict_to_save['CFC12 sat err'].append(sigma_cfc12)
    dict_to_save['SF6 sat err'].append(sigma_sf6)

QSDWARF_TTs = pd.DataFrame(dict_to_save)
QSDWARF_TTs['Station'] = vacao_data_MC_QSDWARF['Station'].values
QSDWARF_TTs['Depth'] = vacao_data_MC_QSDWARF['Depth'].values
QSDWARF_TTs.to_csv('data/QSDWARF_TTs.csv', index=False)
# %% plotting saturations
fig, ax = plt.subplots(2, 4, figsize=(15, 9), sharey=True)

for i, g in enumerate(['CFC12', 'SF6']):
    for j, m in enumerate([TaylorSWIF_TTs, TaylorSWIFT_TTs, DWARF_TTs, QSDWARF_TTs]):
        for k, s in enumerate(vacao_data['Station'].unique()):
            data_to_plot = m[vacao_data_MC_TaylorSWIF['Station'] == s]
            depths_to_plot = vacao_data_MC_TaylorSWIF['Depth'][vacao_data_MC_TaylorSWIF['Station'] == s]
            ax[i][j].errorbar(data_to_plot[g + ' sat']*100, -depths_to_plot, xerr=data_to_plot[g + ' sat err']*100, fmt='.', c=vacao_colors[s], capsize=4, label=f'{s}')
        ax[0][j].set_title(['TaylorSWIF', 'TaylorSWIFT', 'DWARF', 'QSDWARF'][j], fontsize=16)
        ax[i][j].set_xlabel('$\\Delta$' + g + ' [\%]', fontsize=16)
    ax[i, 0].set_ylabel('Depth [m]', fontsize=16)
handles, labels = ax[0, 3].get_legend_handles_labels()
ax[0, 3].legend(handles, labels, fontsize=12, loc='best')


plt.tight_layout()
plt.savefig('plots/paper/SF6 and CFC12 saturations MC.svg', dpi=300, bbox_inches='tight')
plt.show()

CFC12_Results = pd.DataFrame({
    'Station': vacao_data_MC_TaylorSWIF['Station'].astype(int),
    'Depth': vacao_data_MC_TaylorSWIF['Depth'].astype(int),
    'TaylorSWIF': TaylorSWIF_TTs['CFC12 sat'],
    'TaylorSWIF err':  TaylorSWIF_TTs['CFC12 sat err'],
    'TaylorSWIFT': TaylorSWIFT_TTs['CFC12 sat'],
    'TaylorSWIFT err': TaylorSWIFT_TTs['CFC12 sat err'],
    'DWARF': DWARF_TTs['CFC12 sat'],
    'DWARF err': DWARF_TTs['CFC12 sat err'],
    'QSDWARF': QSDWARF_TTs['CFC12 sat'],
    'QSDWARF err': QSDWARF_TTs['CFC12 sat err']
})
CFC12_Results.to_csv('data/CFC12_Results.csv', index=False)

SF6_Results = pd.DataFrame({
    'Station': vacao_data_MC_TaylorSWIF['Station'].astype(int),
    'Depth': vacao_data_MC_TaylorSWIF['Depth'].astype(int),
    'TaylorSWIF': TaylorSWIF_TTs['SF6 sat'],
    'TaylorSWIF err': TaylorSWIF_TTs['SF6 sat err'],
    'TaylorSWIFT': TaylorSWIFT_TTs['SF6 sat'],
    'TaylorSWIFT err': TaylorSWIFT_TTs['SF6 sat err'],
    'DWARF': DWARF_TTs['SF6 sat'],
    'DWARF err': DWARF_TTs['SF6 sat err'],
    'QSDWARF': QSDWARF_TTs['SF6 sat'],
    'QSDWARF err': QSDWARF_TTs['SF6 sat err']
})
SF6_Results.to_csv('data/SF6_Results.csv', index=False)
# %%

### Forward modelling to calculate gas concentrations

# ----- TaylorSWIF --------

gases = ['He', 'Ne', 'Ar', 'Kr', 'Xe']
model_concs_all = []

for i, row in tqdm(vacao_data_MC_TaylorSWIF.iterrows(), total=len(vacao_data_MC_TaylorSWIF)):
    Sample_idx = row['Sample_idx']
    Station = row['Station']
    Depth = row['Depth']
    T_r = float(row['Classical T_r'])   
    sigma_T_r = float(row['MC T_r std'])    
    A = float(row['Classical A'])
    sigma_A = float(row['MC A std'])
    R = float(row['Classical R'])
    sigma_R = float(row['MC R std'])
    S = float(row['S'])
    T = float(row['T'])
    p = 1

    # forward model concentrations
    model_conc = taylor_swif(gases, T_r, S, p, R, A)
    model_errors = derivatives.calc_sigma_TaylorSWIF(gases, T_r, S, p, R, A, sigma_T_r, sigma_R, sigma_A)

    row_values = [Sample_idx, Station, Depth] + list(model_conc) + list(model_errors)
    model_concs_all.append(row_values)

# DataFrame 
columns = ['Sample_idx', 'Station', 'Depth'] + [f'Model_{g}' for g in gases] + [f'Model_{g} err' for g in gases]
model_df_ts = pd.DataFrame(model_concs_all, columns= columns)
model_df_ts.to_csv('data/model_concentrations_TAYLORSWIF.csv', index=False)

# ------- TaylorSWIFT -------

model_concs_all = []

for i, row in tqdm(vacao_data_MC_TaylorSWIFT.iterrows(), total=len(vacao_data_MC_TaylorSWIFT)):
    Sample_idx = row['Sample_idx']
    Station = row['Station']
    Depth = row['Depth']
    T_r = float(row['Classical T_r']   )       
    sigma_T_r = float(row['MC T_r std'])    
    A = float(row['Classical A'])
    sigma_A = float(row['MC A std'])
    R = float(row['Classical R'])
    sigma_R = float(row['MC R std'])
    S = float(row['S'])
    T= float(row['T'])
    p = 1

    # forward model concentrations
    model_conc = taylor_swift(gases, Q(T_r, 'degC'), S, p, R, A)
    model_errors = derivatives.calc_sigma_TaylorSWIFT(gases, T_r, S, p, R, A, sigma_T_r, sigma_R, sigma_A)

    row_values = [Sample_idx, Station, Depth] + list(model_conc) + list(model_errors)
    model_concs_all.append(row_values)

# DataFrame 
columns = ['Sample_idx', 'Station', 'Depth'] + [f'Model_{g}' for g in gases] + [f'Model_{g} err' for g in gases]
model_df_ts = pd.DataFrame(model_concs_all, columns= columns)
model_df_ts.to_csv('data/model_concentrations_TAYLORSWIFT.csv', index=False)



# -------- DWARF -----------
model_concs_all = []
for i, row in tqdm(vacao_data_MC_DWARF.iterrows(), total=len(vacao_data_MC_DWARF)):
    Sample_idx = row['Sample_idx']
    Station = row['Station']
    Depth = row['Depth']
    T_r = float(row['Classical T_r'])   
    sigma_T_r = float(row['MC T_r std'])    
    omega = float(row['Classical omega'])
    sigma_omega = float(row['MC omega std'])
    zeta = float(row['Classical zeta'])
    sigma_zeta = float(row['MC zeta std'])
    S = float(row['S'])
    T = float(row['T'])
    p = 1

    # forward model concentrations
    model_conc = dwarf(gases, T_r, S, p, omega, zeta)
    model_errors = derivatives.calc_sigma_DWARF(gases, T_r, S, p, omega, zeta, sigma_T_r, sigma_omega, sigma_zeta)


    row_values = [Sample_idx, Station, Depth] + list(model_conc) + list(model_errors)
    model_concs_all.append(row_values)
# DataFrame 
columns = ['Sample_idx', 'Station', 'Depth'] + [f'Model_{g}' for g in gases] + [f'Model_{g} err' for g in gases]
model_df_dwarf = pd.DataFrame(model_concs_all, columns= columns)
model_df_dwarf.to_csv('data/model_concentrations_DWARF.csv', index=False)


# -------- QSDWARF -----------
model_concs_all = []
for i, row in tqdm(vacao_data_MC_QSDWARF.iterrows(), total=len(vacao_data_MC_QSDWARF)):
    Sample_idx = row['Sample_idx']
    Station = row['Station']
    Depth = row['Depth']
    T_r = float(row['Classical T_r']) 
    sigma_T_r = float(row['MC T_r std'])     
    omega = float(row['Classical omega'])
    sigma_omega = float(row['MC omega std'])
    zeta = float(row['Classical zeta'])
    sigma_zeta = float(row['MC zeta std'])
    S = float(row['S'])
    T = float(row['T'])
    p = 1

    # forward model concentrations
    model_conc = qs_dwarf(gases, T, S, p, omega, zeta,T_r)
    model_errors = derivatives.calc_sigma_QSDWARF(gases, T, S, p, omega, zeta, T_r, sigma_omega, sigma_zeta, sigma_T_r)
    
    row_values = [Sample_idx, Station, Depth] + list(model_conc) + list(model_errors)
    model_concs_all.append(row_values)
    
# DataFrame 
columns = ['Sample_idx', 'Station', 'Depth'] + [f'Model_{g}' for g in gases] + [f'Model_{g} err' for g in gases]
model_df_dwarf.to_csv('data/model_concentrations_QSDWARF.csv', index=False)



### calculate Chi^2
# ------- TaylorSWIF -------

modeled_df = pd.read_csv('data/model_concentrations_TAYLORSWIF.csv')
measured_df = vacao_data_raw.copy()

# sort df
measured_df = measured_df.sort_values(by=['Station', 'Depth']).reset_index(drop=True)
modeled_df = modeled_df.sort_values(by=['Station', 'Depth']).reset_index(drop=True)

merged_df = pd.merge(measured_df, modeled_df, on=['Station', 'Depth'], suffixes=('', '_Model'))

gases = ['He', 'Ne', 'Ar', 'Kr', 'Xe']
n_params = 3          # T_r, R, A
n_obs = len(gases)    # 5 gases
dof = n_obs - n_params

#calculate  chi^2 
def compute_chi2(row):
    chi2 = 0
    for g in gases:
        model = row[f'Model_{g}']
        obs = row[g]
        err = row[f'{g} err']
        if pd.notnull(model) and pd.notnull(obs) and pd.notnull(err) and err != 0:
            chi2 += ((model - obs) / err) ** 2
    return chi2


merged_df['chi2'] = merged_df.apply(compute_chi2, axis=1)
merged_df['chi2_reduced'] = merged_df['chi2'] / dof
merged_df.to_csv('data/chi2_evaluation_TAYLORSWIF.csv', index=False)


# ------- TaylorSWIFT -------

modeled_df = pd.read_csv('data/model_concentrations_TAYLORSWIFT.csv')

# sort df
measured_df = measured_df.sort_values(by=['Station', 'Depth']).reset_index(drop=True)
modeled_df = modeled_df.sort_values(by=['Station', 'Depth']).reset_index(drop=True)

merged_df = pd.merge(measured_df, modeled_df, on=['Station', 'Depth'], suffixes=('', '_Model'))

gases = ['He', 'Ne', 'Ar', 'Kr', 'Xe']
n_params = 3          # T_r, omega, zeta
n_obs = len(gases)    # 5 Gase
dof = n_obs - n_params


merged_df['chi2'] = merged_df.apply(compute_chi2, axis=1)
merged_df['chi2_reduced'] = merged_df['chi2'] / dof
merged_df.to_csv('data/chi2_evaluation_TAYLORSWIFT.csv', index=False)

# ------- DWARF -------

modeled_df = pd.read_csv('data/model_concentrations_DWARF.csv')

# sort df
measured_df = measured_df.sort_values(by=['Station', 'Depth']).reset_index(drop=True)
modeled_df = modeled_df.sort_values(by=['Station', 'Depth']).reset_index(drop=True)

merged_df = pd.merge(measured_df, modeled_df, on=['Station', 'Depth'], suffixes=('', '_Model'))

gases = ['He', 'Ne', 'Ar', 'Kr', 'Xe']
n_params = 3          # T_r, omega, zeta
n_obs = len(gases)    # 5 Gase
dof = n_obs - n_params

# calculate chi^2 
merged_df['chi2'] = merged_df.apply(compute_chi2, axis=1)
merged_df['chi2_reduced'] = merged_df['chi2'] / dof
merged_df.to_csv('data/chi2_evaluation_DWARF.csv', index=False)


# ------- QSDWARF -------

modeled_df = pd.read_csv('data/model_concentrations_QSDWARF.csv')

# sort df
measured_df = measured_df.sort_values(by=['Station', 'Depth']).reset_index(drop=True)
modeled_df = modeled_df.sort_values(by=['Station', 'Depth']).reset_index(drop=True)

merged_df = pd.merge(measured_df, modeled_df, on=['Station', 'Depth'], suffixes=('', '_Model'))

gases = ['He', 'Ne', 'Ar', 'Kr', 'Xe']
n_params = 3          # T_r, omega, zeta
n_obs = len(gases)    # 5 Gase
dof = n_obs - n_params

# calculate chi^2 

merged_df['chi2'] = merged_df.apply(compute_chi2, axis=1)
merged_df['chi2_reduced'] = merged_df['chi2'] / dof
merged_df.to_csv('data/chi2_evaluation_QSDWARF.csv', index=False)

# %%
# plot