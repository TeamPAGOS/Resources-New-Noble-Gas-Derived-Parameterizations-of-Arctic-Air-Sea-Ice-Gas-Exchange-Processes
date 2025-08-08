import numpy as np
import pandas as pd
from pagos import Q
from pagos import builtin_models as pbim
from pagos.modelling import GasExchangeModel
from datetime import datetime
import os
from tqdm import tqdm
rng = np.random.default_rng()
pd.set_option('display.float_format', '{:.10e}'.format)
import matplotlib.pyplot as plt
import glob
import seaborn as sns
from scipy.stats import norm
from matplotlib.backends.backend_pdf import PdfPages
from pagos import u




df = pd.read_csv('data/NG_data_all_avg_final_adjusted.CSV', sep=',')

#Monte Carlo Set Up
n_simulations = 10000
gases_used = ['He','Ne', 'Ar', 'Kr', 'Xe'] 


#---------------------#
#  Taylor Swif Model  #
#---------------------#

taylor_swif_model = pbim.taylor_swif
TAYLORSWIFModel = GasExchangeModel(taylor_swif_model, ('degC', 'permille', 'atm','', 'cc/g'), 'cc/g')

# output folder 
output_dir = "WRITE_YOUR_OUTPUT_DIRECTORY_PATH_HERE_SPECIFIC_TO_TAYLOR_SWIF"
os.makedirs(output_dir, exist_ok=True)
# Collect results per sample

for probe_index in tqdm(range(len(df))):
    station_data = df.iloc[probe_index]

    gas_concentrations = []
    gas_errors = []
    gases_available = []

    for gas in gases_used:
        gas_val = station_data.get(f'{gas}', None)
        gas_err = station_data.get(f'{gas} err', None)
        
        if gas_val is not None and gas_err is not None:
            gas_concentrations.append(gas_val)
            gas_errors.append(gas_err)
            gases_available.append(gas)

    if len(gases_available) == 0:
        # skip faulty samples
        print(f"No gases found for {probe_index}, skip.")
        continue
    
    gas_units = np.array(['cc/g'] * len(gases_available))  
    tuple_errors = gas_errors 
    
    obs_params = [station_data['S'],station_data['p']]  

    probe_fit_results = []

    # Simulations
    mc_simulations = np.random.normal(loc=gas_concentrations, scale=gas_errors, size=(n_simulations, len(gases_available)))
    T0, A0, R0 = TAYLORSWIFModel.fit((gas_concentrations, gas_errors, 'cc/g', obs_params), ['T_r', 'A', 'R'], [0, 1e-5, 0.1], gases_available)
    for i in range(n_simulations):
        simulated_tracers = mc_simulations[i]

        data_tuple = (
            simulated_tracers,  # obs_tracers
            tuple_errors,         
            gas_units,          
            obs_params          
        )

        fitted = TAYLORSWIFModel.fit(data_tuple,
                             to_fit=['T_r', 'A', 'R'],
                             init_guess=[Q(273.15, 'K'), 1e-5, 0.1],
                             tracers_used=gases_available,
                             tqdm_bar=True)

        T_fit, A_fit, R_fit = fitted
        probe_fit_results.append([T_fit.nominal_value, A_fit.nominal_value, R_fit.nominal_value])  

    # DataFrame with fit results
    results_df = pd.DataFrame(probe_fit_results, columns=['T_r', 'A', 'R'])
    results_df.insert(0, "realization", range(1, n_simulations + 1))

    # Shared file names for better allocation
    today = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_filename = f"MC_Sim_TAYLORSWIF_Sample_{probe_index}_{today}"
    
    # Output 
    header_filename = os.path.join(output_dir, f"{base_filename}_HEADER.csv")
    data_filename = os.path.join(output_dir, f"{base_filename}.pkl.gz")

    # save metadata as CSV
    with open(header_filename, 'w') as f:
        f.write("Monte Carlo Sim Header Info\n")
        f.write(f"Station,{station_data['Station']}\n")
        f.write(f"Depth,{station_data['Depth']}\n")

        for gas, val, err in zip(gases_available, gas_concentrations, gas_errors):
            f.write(f"{gas},{val:.5e}\n")
            f.write(f"{gas}_err,{err:.5e}\n")

        f.write(f"S,{station_data['S']}\n")
        f.write(f"p,{station_data['p']}\n")
        f.write("Columns," + ",".join(results_df.columns) + "\n")

    # save as compressed pickle
    results_df.to_pickle(data_filename, compression="gzip")



#---------------------#
#  Taylor Swift Model  #
#---------------------#

taylor_swift_model = pbim.taylor_swift
TAYLORSWIFTModel = GasExchangeModel(taylor_swift_model, ('degC', 'permille', 'atm','', 'cc/g'), 'cc/g')


# output file directory 
output_dir = "WRITE_YOUR_OUTPUT_DIRECTORY_PATH_HERE_SPECIFIC_TO_TAYLOR_SWIFT"
os.makedirs(output_dir, exist_ok=True)


for probe_index in tqdm(range(len(df))):
    station_data = df.iloc[probe_index]

    gas_concentrations = []
    gas_errors = []
    gases_available = []

    for gas in gases_used:
        gas_val = station_data.get(f'{gas}', None)
        gas_err = station_data.get(f'{gas} err', None)
        
        if gas_val is not None and gas_err is not None:
            gas_concentrations.append(gas_val)
            gas_errors.append(gas_err)
            gases_available.append(gas)

    if len(gases_available) == 0:
        print(f"No gases found for {probe_index}, skip.")
        continue
    
    gas_units = np.array(['cc/g'] * len(gases_available))  
    tuple_errors = gas_errors 
    obs_params = [station_data['S'],station_data['p']]  

    probe_fit_results = []

    # Simulations 
    mc_simulations = np.random.normal(loc=gas_concentrations, scale=gas_errors, size=(n_simulations, len(gases_available)))

    for i in range(n_simulations):
        simulated_tracers = mc_simulations[i]

        data_tuple = (
            simulated_tracers,  # obs_tracers
            tuple_errors,         
            gas_units,          
            obs_params          
        )

        fitted = TAYLORSWIFTModel.fit(data_tuple,
                             to_fit=['T_r', 'A', 'R'],
                             init_guess=[Q(273.15, 'K'), 1e-5, 0.1],
                             tracers_used=gases_available,
                             tqdm_bar=True)

        T_fit, A_fit, R_fit = fitted
        probe_fit_results.append([T_fit.nominal_value, A_fit.nominal_value, R_fit.nominal_value])  

    # DataFrame with fit results
    results_df = pd.DataFrame(probe_fit_results, columns=['T_r', 'A', 'R'])
    results_df.insert(0, "realization", range(1, n_simulations + 1))

    # shared file names for better allocation
    today = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_filename = f"MC_Sim_TAYLORSWIFT_Sample_{probe_index}_{today}"
    
    # output paths
    header_filename = os.path.join(output_dir, f"{base_filename}_HEADER.csv")
    data_filename = os.path.join(output_dir, f"{base_filename}.pkl.gz")

    # save metadata as CSV
    with open(header_filename, 'w') as f:
        f.write("Monte Carlo Sim Header Info\n")
        f.write(f"Station,{station_data['Station']}\n")
        f.write(f"Depth,{station_data['Depth']}\n")

        for gas, val, err in zip(gases_available, gas_concentrations, gas_errors):
            f.write(f"{gas},{val:.5e}\n")
            f.write(f"{gas}_err,{err:.5e}\n")

        f.write(f"S,{station_data['S']}\n")
        f.write(f"p,{station_data['p']}\n")
        f.write("Columns," + ",".join(results_df.columns) + "\n")

    # save as compressed pickle
    results_df.to_pickle(data_filename, compression="gzip")

#---------------------#
#  DWARF Model  #
#---------------------#

dwarf_model = pbim.dwarf
DWARFModel = GasExchangeModel(dwarf_model, ('degC', 'permille', 'atm','', 'cc/g'), 'cc/g')


# output directory
output_dir = "WRITE_YOUR_OUTPUT_DIRECTORY_PATH_HERE_SPECIFIC_TO_DWARF"
os.makedirs(output_dir, exist_ok=True)


for probe_index in tqdm(range(len(df))):
    station_data = df.iloc[probe_index]

    gas_concentrations = []
    gas_errors = []
    gases_available = []

    for gas in gases_used:
        gas_val = station_data.get(f'{gas}', None)
        gas_err = station_data.get(f'{gas} err', None)
        
        if gas_val is not None and gas_err is not None:
            gas_concentrations.append(gas_val)
            gas_errors.append(gas_err)
            gases_available.append(gas)

    if len(gases_available) == 0:
    
        print(f"No gases found for {probe_index}, skip.")
        continue
    
    gas_units = np.array(['cc/g'] * len(gases_available))  
    tuple_errors = gas_errors 
    obs_params = [station_data['S'],station_data['p']]  

    probe_fit_results = []

    # Simulations
    mc_simulations = np.random.normal(loc=gas_concentrations, scale=gas_errors, size=(n_simulations, len(gases_available)))

    for i in range(n_simulations):
        simulated_tracers = mc_simulations[i]

        data_tuple = (
            simulated_tracers,  # obs_tracers
            tuple_errors,         
            gas_units,          
            obs_params          
        )

        fitted = DWARFModel.fit(data_tuple,
                             to_fit=['T_r', 'zeta', 'omega'],
                             init_guess=[Q(273.15, 'K'), 1e-5, 0.1],
                             tracers_used=gases_available,
                             tqdm_bar=True)

        T_fit, zeta_fit, omega_fit = fitted
        probe_fit_results.append([T_fit.nominal_value, zeta_fit.nominal_value, omega_fit.nominal_value])  

    # DataFrame with fit results
    results_df = pd.DataFrame(probe_fit_results, columns=['T_r', 'zeta', 'omega'])
    results_df.insert(0, "realization", range(1, n_simulations + 1))

    # shared file names for better allocation
    today = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_filename = f"MC_Sim_DWARF_Sample_{probe_index}_{today}"
    
    # output paths 
    header_filename = os.path.join(output_dir, f"{base_filename}_HEADER.csv")
    data_filename = os.path.join(output_dir, f"{base_filename}.pkl.gz")

    # save metadata as CSV
    with open(header_filename, 'w') as f:
        f.write("Monte Carlo Sim Header Info\n")
        f.write(f"Station,{station_data['Station']}\n")
        f.write(f"Depth,{station_data['Depth']}\n")

        for gas, val, err in zip(gases_available, gas_concentrations, gas_errors):
            f.write(f"{gas},{val:.5e}\n")
            f.write(f"{gas}_err,{err:.5e}\n")

        f.write(f"S,{station_data['S']}\n")
        f.write(f"p,{station_data['p']}\n")
        f.write("Columns," + ",".join(results_df.columns) + "\n")

    # save as compressed pickle
    results_df.to_pickle(data_filename, compression="gzip")




#---------------------#
#  QS-DWARF Model  #
#---------------------#

qs_dwarf_model = pbim.qs_dwarf
QSDWARFModel = GasExchangeModel(qs_dwarf_model, ('degC', 'permille', 'atm','', 'cc/g', 'degC'), 'cc/g')


# output directory 
output_dir = "WRITE_YOUR_OUTPUT_DIRECTORY_PATH_HERE_SPECIFIC_TO_QSDWARF"
os.makedirs(output_dir, exist_ok=True)


for probe_index in tqdm(range(len(df))):
    station_data = df.iloc[probe_index]

    gas_concentrations = []
    gas_errors = []
    gases_available = []

    for gas in gases_used:
        gas_val = station_data.get(f'{gas}', None)
        gas_err = station_data.get(f'{gas} err', None)
        
        if gas_val is not None and gas_err is not None:
            gas_concentrations.append(gas_val)
            gas_errors.append(gas_err)
            gases_available.append(gas)

    if len(gases_available) == 0:
        print(f"No gases found for {probe_index}, skip.")
        continue
    
    gas_units = np.array(['cc/g'] * len(gases_available))  
    tuple_errors = gas_errors 
    obs_params = [station_data['T'],station_data['S'],station_data['p']]  

    probe_fit_results = []

    # Simulations
    mc_simulations = np.random.normal(loc=gas_concentrations, scale=gas_errors, size=(n_simulations, len(gases_available)))

    for i in range(n_simulations):
        simulated_tracers = mc_simulations[i]

        data_tuple = (
            simulated_tracers,  # obs_tracers
            tuple_errors,         
            gas_units,          
            obs_params          
        )

        fitted = QSDWARFModel.fit(data_tuple,
                             to_fit=['T_r', 'zeta', 'omega'],
                             init_guess=[Q(273.15, 'K'), 1e-5, 0.1],
                             tracers_used=gases_available,
                             tqdm_bar=True)

        T_r_fit, zeta_fit, omega_fit = fitted
        probe_fit_results.append([T_r_fit.nominal_value, zeta_fit.nominal_value, omega_fit.nominal_value])  

    # DataFrame with fit results
    results_df = pd.DataFrame(probe_fit_results, columns=['T_r', 'zeta', 'omega'])
    results_df.insert(0, "realization", range(1, n_simulations + 1))

    # shared file names for better allocation
    today = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_filename = f"MC_Sim_QSDWARF_Sample_{probe_index}"
    
    # output paths 
    header_filename = os.path.join(output_dir, f"{base_filename}_HEADER.csv")
    data_filename = os.path.join(output_dir, f"{base_filename}.pkl.gz")

    # save metadata as CSV
    with open(header_filename, 'w') as f:
        f.write("Monte Carlo Sim Header Info\n")
        f.write(f"Station,{station_data['Station']}\n")
        f.write(f"Depth,{station_data['Depth']}\n")

        for gas, val, err in zip(gases_available, gas_concentrations, gas_errors):
            f.write(f"{gas},{val:.5e}\n")
            f.write(f"{gas}_err,{err:.5e}\n")

        f.write(f"S,{station_data['S']}\n")
        f.write(f"p,{station_data['p']}\n")
        f.write("Columns," + ",".join(results_df.columns) + "\n")

    # save as compressed pickle
    results_df.to_pickle(data_filename, compression="gzip")



#---------------------#
#  Plotting histograms  #
#---------------------#



today_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')



# Basis-Pfade anpassen
base_path = 'PATH/TO/YOUR/MC/OUTPUT/FOR/TaylorSWIF/...'                     # MC data
classical_csv_path = '/PATH/TO/YOUR/CLASSICAL/FIT/DATA/FOR/TaylorSWIF/...'  #classical fits
metadata_path = 'PATH/TO/HEADERS/FOR/TaylorSWIF/...'                        # MC metadata
output_dir = "OUTPUT/IMAGES/DIRECTORY/"                                     #output directory
os.makedirs(output_dir, exist_ok=True)


# find all Pickle files 
pkl_files = sorted(glob.glob(os.path.join(base_path, "MC_Sim_TAYLORSWIF_Sample_*.pkl.gz")))
print(f"Anzahl gefundener Pickle-Dateien: {len(pkl_files)}")


# load metadata
metadata_files = sorted(glob.glob(os.path.join(metadata_path, "MC_Sim_TAYLORSWIF_Sample_*_HEADER.csv")))


# load classical values
classical_df = pd.read_csv(classical_csv_path)
metadata_map = {
    int(os.path.basename(f).split("_Sample_")[1].split("_")[0]): f
    for f in metadata_files
}

# colors
color = sns.color_palette("husl")[0]

results = []

for pkl in pkl_files:
    # sample number from file name
    sample_str = os.path.basename(pkl).split("_Sample_")[1].split("_")[0]
    sample_idx = int(sample_str)

    # ensure that a header file exists
    if sample_idx not in metadata_map:
        print(f"no header for sample {sample_idx}, skip.")
        continue

    metadata_file = metadata_map[sample_idx]

    # header file loading
    with open(metadata_file, 'r') as f:
        header_lines = f.readlines()

    header_dict = {}
    for line in header_lines:
        parts = line.strip().split(",")
        if len(parts) == 2:
            key, value = parts
            header_dict[key] = value

    station = header_dict.get("Station", "NA")
    depth = header_dict.get("Depth", "NA")
    S = header_dict.get("S", "NA")

    # load MC data
    df_sim = pd.read_pickle(pkl, compression='infer')
    df_sim.columns = ['d', 'T_r', 'A', 'R']


    t_r_val = classical_df.loc[sample_idx, 'T_r']
    t_r_err = classical_df.loc[sample_idx, 'T_r err']
    a_val = classical_df.loc[sample_idx, 'A']
    a_err = classical_df.loc[sample_idx, 'A err']
    r_val = classical_df.loc[sample_idx, 'R']
    r_err = classical_df.loc[sample_idx, 'R err']

    klassisch_values = {'T_r': (t_r_val, t_r_err), 'A': (a_val, a_err), 'R': (r_val, r_err)}

    # Gaussian-Fits
    mc_fit = {}
    for param in ['T_r', 'A', 'R']:
        data = df_sim[param].dropna()
        mu, std = norm.fit(data)
        mc_fit[param] = (mu, std)


    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    for ax, param in zip(axes, ['T_r', 'A', 'R']):
        data = df_sim[param].dropna()
        einheiten = {
            'T_r': '°C',
            'A': 'cc/g',
            'R': '-'
        }

        sns.histplot(data, bins=30, kde=False, stat='density', element="step", color='blue', ax=ax)

        mu, std = mc_fit[param]
        x = np.linspace(data.min(), data.max(), 1000)
        p = norm.pdf(x, mu, std)
        label = (f"MC" if param == 'T_r' else f"MC: {mu:.2e} ± {std:.2e}")
        ax.plot(x, p, 'k--', linewidth=2, label=label)
        ax.axvline(mu - std, color='black', linestyle='--')
        ax.axvline(mu + std, color='black', linestyle='--')

        val, err = klassisch_values[param]
        ax.axvspan(val - err, val + err, color='red', alpha=0.3, label="Classical error")
        label = (f"Classical" if param == 'T_r' else f"Classical: {val:.2e} ± {err:.2e}")
        ax.axvline(val, color='black', linewidth=2, label=label)

        greek_labels = {
            'T_r': r"$T_r$",
            'A': r"$A$",
            'R': r"$R$"
        }
    
        ax.set_xlabel(f"{greek_labels[param]} [{einheiten.get(param, '')}]", fontsize=20)
        ax.set_ylabel("Density", fontsize=20)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)  
        ax.tick_params(labelsize=18)
        
        if param == 'R':
            ax.legend(fontsize=12, loc='upper right')


    # Station and Depth as floats
    try:
        station_int = int(float(station))  
    except ValueError:
        station_int = station  

    try:
        depth_int = int(float(depth))  
    except ValueError:
        depth_int = depth

    plt.suptitle(f"TaylorSWIF: \nStation {station_int} / Depth {depth_int} m", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
   

    # save
    filename = os.path.join(
        output_dir,
        f"Vergleich_TAYLORSWIF_AlleParameter_Station{station_int}_Depth{depth_int}_Sample_{sample_str}_{today_str}.pdf"
    )
    plt.savefig(filename, dpi=1200, format='pdf') 
    plt.close()

    # save results per sample
    results.append({
        'Sample_idx': sample_idx,
        'Station': station,
        'Depth': depth,
        'Classical T_r': klassisch_values['T_r'][0],
        'Classical T_r err': klassisch_values['T_r'][1],
        'MC T_r': mc_fit['T_r'][0],
        'MC T_r std': mc_fit['T_r'][1],
        'Classical A': klassisch_values['A'][0],
        'Classical A err': klassisch_values['A'][1],
        'MC A': mc_fit['A'][0],
        'MC A std': mc_fit['A'][1],
        'Classical R': klassisch_values['R'][0],
        'Classical R err': klassisch_values['R'][1],
        'MC R': mc_fit['R'][0],
        'MC R std': mc_fit['R'][1],
        'S': S
    })

#save results to a CSV file
output_file = os.path.join(output_dir, f"Comparison_Results_TAYLORSWIF.csv")
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)



#---------------------#
#  Taylor SWIFT Model  #
#---------------------#

# Base path
base_path = 'PATH/TO/YOUR/MC/OUTPUT/FOR/TaylorSWIFT/...'                     # MC data
classical_csv_path = '/PATH/TO/YOUR/CLASSICAL/FIT/DATA/FOR/TaylorSWIFT/...'  #classical fits
metadata_path = 'PATH/TO/HEADERS/FOR/TaylorSWIFT/...'                        # MC metadata
output_dir = "OUTPUT/IMAGES/DIRECTORY/"                                      #output directory
os.makedirs(output_dir, exist_ok=True)


# find all Pickle files 
pkl_files = sorted(glob.glob(os.path.join(base_path, "MC_Sim_TAYLORSWIFT_Sample_*.pkl.gz")))
print(f"Anzahl gefundener Pickle-Dateien: {len(pkl_files)}")


# load metadata
metadata_files = sorted(glob.glob(os.path.join(metadata_path, "MC_Sim_TAYLORSWIFT_Sample_*_HEADER.csv")))


# load classical values
classical_df = pd.read_csv(classical_csv_path)
metadata_map = {
    int(os.path.basename(f).split("_Sample_")[1].split("_")[0]): f
    for f in metadata_files
}

# colors
color = sns.color_palette("husl")[0]

results = []

for pkl in pkl_files:
    # get sample number from file name
    sample_str = os.path.basename(pkl).split("_Sample_")[1].split("_")[0]
    sample_idx = int(sample_str)

    # make sure a header file exists
    if sample_idx not in metadata_map:
        print(f"Kein Header für Sample {sample_idx}, übersprungen.")
        continue

    metadata_file = metadata_map[sample_idx]

    # load header file
    with open(metadata_file, 'r') as f:
        header_lines = f.readlines()

    header_dict = {}
    for line in header_lines:
        parts = line.strip().split(",")
        if len(parts) == 2:
            key, value = parts
            header_dict[key] = value

    station = header_dict.get("Station", "NA")
    depth = header_dict.get("Depth", "NA")
    S = header_dict.get("S", "NA")
  

    # load MC data
    df_sim = pd.read_pickle(pkl, compression='infer')
    df_sim.columns = ['d', 'T_r', 'A', 'R']


    t_r_val = classical_df.loc[sample_idx, 'T_r']
    t_r_err = classical_df.loc[sample_idx, 'T_r err']
    a_val = classical_df.loc[sample_idx, 'A']
    a_err = classical_df.loc[sample_idx, 'A err']
    r_val = classical_df.loc[sample_idx, 'R']
    r_err = classical_df.loc[sample_idx, 'R err']

    klassisch_values = {'T_r': (t_r_val, t_r_err), 'A': (a_val, a_err), 'R': (r_val, r_err)}

    # Gaussian-Fits
    mc_fit = {}
    for param in ['T_r', 'A', 'R']:
        data = df_sim[param].dropna()
        mu, std = norm.fit(data)
        mc_fit[param] = (mu, std)

   
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, param in zip(axes, ['T_r', 'A', 'R']):
        data = df_sim[param].dropna()
        einheiten = {
            'T_r': '°C',
            'A': 'cc/g',
            'R': ''
        }

        sns.histplot(data, bins=30, kde=False, stat='density', element="step", color='blue', ax=ax)

        mu, std = mc_fit[param]
        x = np.linspace(data.min(), data.max(), 1000)
        p = norm.pdf(x, mu, std)
        label = (f"Gaussian-Fit: {mu:.2f} ± {std:.2f}" if param == 'T_r' 
                 else f"Gaussian Fit: {mu:.2e} ± {std:.2e}")
        ax.plot(x, p, 'k--', linewidth=2, label=label)

        val, err = klassisch_values[param]
        ax.axvspan(val - err, val + err, color='red', alpha=0.3, label="Error range")
        label = (f"simple Fit: {val:.2f} ± {err:.2f}" if param == 'T_r'
                 else f"simple Fit: {val:.2e} ± {err:.2e}")
        ax.axvline(val, color='black', linewidth=2, label=label)

        einheit = einheiten.get(param, '')
        ax.set_xlabel(f"{param} [{einheit}]" if einheit else param, fontsize=18)
        ax.set_ylabel("Density", fontsize=18)
        ax.set_title(f"{param}", fontsize=18)
        ax.tick_params(labelsize=16)
        ax.legend(fontsize=14, loc='upper right')

    # Station and Depth as floats 
    try:
        station_int = int(float(station))  
    except ValueError:
        station_int = station  

    try:
        depth_int = int(float(depth))  
    except ValueError:
        depth_int = depth

    plt.suptitle(f"Sample {sample_str}: \nStation {station_int} / Depth {depth_int} m", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # saving
    filename = os.path.join(
        output_dir,
        f"Vergleich_TAYLORSWIFT_AlleParameter_Station{station_int}_Depth{depth_int}_Sample_{sample_str}_{today_str}.pdf"
    )
    plt.savefig(filename, dpi=1200, format='pdf')
    plt.close()

    # save results per sample
    results.append({
        'Sample_idx': sample_idx,
        'Station': station,
        'Depth': depth,
        'Classical T_r': klassisch_values['T_r'][0],
        'Classical T_r err': klassisch_values['T_r'][1],
        'MC T_r': mc_fit['T_r'][0],
        'MC T_r std': mc_fit['T_r'][1],
        'Classical A': klassisch_values['A'][0],
        'Classical A err': klassisch_values['A'][1],
        'MC A': mc_fit['A'][0],
        'MC A std': mc_fit['A'][1],
        'Classical R': klassisch_values['R'][0],
        'Classical R err': klassisch_values['R'][1],
        'MC R': mc_fit['R'][0],
        'MC R std': mc_fit['R'][1],
        'S': S
    })

#save results to a CSV file
output_file = os.path.join(output_dir, f"Comparison_Results_TAYLORSWIFT.csv")
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)


### ----------- ###
### DWARF Model ###
### ----------- ###


# base paths
base_path = 'PATH/TO/YOUR/MC/OUTPUT/FOR/DWARF/...'                     # MC data
classical_csv_path = '/PATH/TO/YOUR/CLASSICAL/FIT/DATA/FOR/DWARF/...'  #classical fits
metadata_path = 'PATH/TO/HEADERS/FOR/DWARF/...'                        # MC metadata
output_dir = "OUTPUT/IMAGES/DIRECTORY/"                                #output directory
os.makedirs(output_dir, exist_ok=True)

# find all Pickle files
pkl_files = sorted(glob.glob(os.path.join(base_path, "MC_Sim_DWARF_Sample_*.pkl.gz")))

# get metadata
metadata_files = sorted(glob.glob(os.path.join(metadata_path, "MC_Sim_DWARF_Sample_*_HEADER.csv")))


# get classical values
classical_df = pd.read_csv(classical_csv_path)
metadata_map = {
    int(os.path.basename(f).split("_Sample_")[1].split("_")[0]): f
    for f in metadata_files
}

# colors
color = sns.color_palette("husl")[0]

results = []
for pkl in pkl_files:
    # get sample number from file name
    sample_str = os.path.basename(pkl).split("_Sample_")[1].split("_")[0]
    sample_idx = int(sample_str)

    if sample_idx not in metadata_map:
        print(f"Kein Header für Sample {sample_idx}, übersprungen.")
        continue

    metadata_file = metadata_map[sample_idx]

    # get header file
    with open(metadata_file, 'r') as f:
        header_lines = f.readlines()

    header_dict = {}
    for line in header_lines:
        parts = line.strip().split(",")
        if len(parts) == 2:
            key, value = parts
            header_dict[key] = value

    station = header_dict.get("Station", "NA")
    depth = header_dict.get("Depth", "NA")
    S = header_dict.get("S", "NA")

    # get MC data
    df_sim = pd.read_pickle(pkl, compression='infer')
    df_sim.columns = ['r', 'T_r', 'zeta', 'omega']

    # classical values from the classical fit
    t_r_val = classical_df.loc[sample_idx, 'T_r']
    t_r_err = classical_df.loc[sample_idx, 'T_r err']
    zeta_val = classical_df.loc[sample_idx, 'zeta']
    zeta_err = classical_df.loc[sample_idx, 'zeta err']
    omega_val = classical_df.loc[sample_idx, 'omega']
    omega_err = classical_df.loc[sample_idx, 'omega err']

    klassisch_values = {'T_r': (t_r_val, t_r_err), 'zeta': (zeta_val, zeta_err), 'omega': (omega_val, omega_err)}
    
    # Gaussian-Fits
    mc_fit = {}
    for param in ['T_r', 'zeta', 'omega']:
        data = df_sim[param].dropna()
        mu, std = norm.fit(data)
        mc_fit[param] = (mu, std)


    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, param in zip(axes, ['T_r', 'omega', 'zeta']):
        data = df_sim[param].dropna()
        einheiten = {
        'T_r': '°C',
        'omega': '',
        'zeta' : 'cc/g'  
        }

        sns.histplot(data, bins=30, kde=False, stat='density', element="step", color='blue', ax=ax)

        mu, std = mc_fit[param]
        x = np.linspace(data.min(), data.max(), 1000)
        p = norm.pdf(x, mu, std)
        label = (f"Gaussian-Fit: {mu:.2f} ± {std:.2f}" if param == 'T_r' 
                 else f"Gaussian Fit: {mu:.2e} ± {std:.2e}")
        ax.plot(x, p, 'k--', linewidth=2, label=label)

        val, err = klassisch_values[param]
        ax.axvspan(val - err, val + err, color='red', alpha=0.3, label="Error range")
        label = (f"simple Fit: {val:.2f} ± {err:.2f}" if param == 'T_r'
                 else f"simple Fit: {val:.2e} ± {err:.2e}")
        ax.axvline(val, color='black', linewidth=2, label=label)

        einheit = einheiten.get(param, '')
        ax.set_xlabel(f"{param} [{einheit}]" if einheit else param, fontsize=18)
        ax.set_ylabel("Density", fontsize=18)
        ax.set_title(f"{param}", fontsize=18)
        ax.tick_params(labelsize=16)
        ax.legend(fontsize=14, loc='upper right')

    # Station and Depth as floats 
    try:
        station_int = int(float(station))  
    except ValueError:
        station_int = station  

    try:
        depth_int = int(float(depth))  
    except ValueError:
        depth_int = depth

    plt.suptitle(f"Sample {sample_str}: \nStation {station_int} / Depth {depth_int} m", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # save
    filename = os.path.join(
        output_dir,
        f"Vergleich_DWARF_AlleParameter_Station{station_int}_Depth{depth_int}_Sample_{sample_str}_{today_str}.pdf"
    )
    plt.savefig(filename, dpi=1200, format='pdf')
    plt.close()

    # save results per sample
    results.append({
        'Sample_idx': sample_idx,
        'Station': station,
        'Depth': depth,
        'Classical T_r': klassisch_values['T_r'][0],
        'Classical T_r err': klassisch_values['T_r'][1],
        'MC T_r': mc_fit['T_r'][0],
        'MC T_r std': mc_fit['T_r'][1],
        'Classical zeta': klassisch_values['zeta'][0],
        'Classical zeta err': klassisch_values['zeta'][1],
        'MC zeta': mc_fit['zeta'][0],
        'MC zeta std': mc_fit['zeta'][1],
        'Classical omega': klassisch_values['omega'][0],
        'Classical omega err': klassisch_values['omega'][1],
        'MC omega': mc_fit['omega'][0],
        'MC omega std': mc_fit['omega'][1],
        'S' : S
    })


#save results to a CSV file
output_file = os.path.join(output_dir, f"Comparison_Results_DWARF.csv")
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)



#---------------------#
#  QSDWARF Model  #
#---------------------#


# base paths
base_path = 'PATH/TO/YOUR/MC/OUTPUT/FOR/QSDWARF/...'                     # MC data
classical_csv_path = '/PATH/TO/YOUR/CLASSICAL/FIT/DATA/FOR/QSDWARF/...'  # classical fits
metadata_path = 'PATH/TO/HEADERS/FOR/QSDWARF/...'                        # MC metadata
output_dir = "OUTPUT/IMAGES/DIRECTORY/"                                  # output directory
os.makedirs(output_dir, exist_ok=True)

# find all Pickle files
pkl_files = sorted(glob.glob(os.path.join(base_path, "MC_Sim_QSDWARF_Sample_*.pkl.gz")))

# get metadata
metadata_files = sorted(glob.glob(os.path.join(metadata_path, "MC_Sim_QSDWARF_Sample_*_HEADER.csv")))


# get classical values
classical_df = pd.read_csv(classical_csv_path)
metadata_map = {
    int(os.path.basename(f).split("_Sample_")[1].split("_")[0]): f
    for f in metadata_files
}

# color
color = sns.color_palette("husl")[0]

results = []
for pkl in pkl_files:
    # get sample number from file name
    sample_str = os.path.basename(pkl).split("_Sample_")[1].split("_")[0]
    sample_idx = int(sample_str)

    if sample_idx not in metadata_map:
        print(f"Kein Header für Sample {sample_idx}, übersprungen.")
        continue

    metadata_file = metadata_map[sample_idx]

    # Metadata
    with open(metadata_file, 'r') as f:
        header_lines = f.readlines()

    header_dict = {}
    for line in header_lines:
        parts = line.strip().split(",")
        if len(parts) == 2:
            key, value = parts
            header_dict[key] = value

    station = header_dict.get("Station", "NA")
    depth = header_dict.get("Depth", "NA")
    S = header_dict.get("S", "NA")

    # MC data
    df_sim = pd.read_pickle(pkl, compression='infer')
    df_sim.columns = ['r', 'T_r', 'zeta', 'omega']

    # classical values from the classical fit
    t_r_val = classical_df.loc[sample_idx, 'T_r']
    t_r_err = classical_df.loc[sample_idx, 'T_r err']
    zeta_val = classical_df.loc[sample_idx, 'zeta']
    zeta_err = classical_df.loc[sample_idx, 'zeta err']
    omega_val = classical_df.loc[sample_idx, 'omega']
    omega_err = classical_df.loc[sample_idx, 'omega err']

    klassisch_values = {'T_r': (t_r_val, t_r_err), 'zeta': (zeta_val, zeta_err), 'omega': (omega_val, omega_err)}
    
    # Gaussian-Fits
    mc_fit = {}
    for param in ['T_r', 'zeta', 'omega']:
        data = df_sim[param].dropna()
        mu, std = norm.fit(data)
        mc_fit[param] = (mu, std)

  
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, param in zip(axes, ['T_r', 'omega', 'zeta']):
        data = df_sim[param].dropna()
        einheiten = {
        'T_r': '°C',
        'omega': '',
        'zeta' : 'cc/g'  
        }

        sns.histplot(data, bins=30, kde=False, stat='density', element="step", color='blue', ax=ax)

        mu, std = mc_fit[param]
        x = np.linspace(data.min(), data.max(), 1000)
        p = norm.pdf(x, mu, std)
        label = (f"Gaussian-Fit: {mu:.2f} ± {std:.2f}" if param == 'T_r' 
                 else f"Gaussian Fit: {mu:.2e} ± {std:.2e}")
        ax.plot(x, p, 'k--', linewidth=2, label=label)

        val, err = klassisch_values[param]
        ax.axvspan(val - err, val + err, color='red', alpha=0.3, label="Error range")
        label = (f"simple Fit: {val:.2f} ± {err:.2f}" if param == 'T_r'
                 else f"simple Fit: {val:.2e} ± {err:.2e}")
        ax.axvline(val, color='black', linewidth=2, label=label)

        einheit = einheiten.get(param, '')
        ax.set_xlabel(f"{param} [{einheit}]" if einheit else param, fontsize=18)
        ax.set_ylabel("Density", fontsize=18)
        ax.set_title(f"{param}", fontsize=18)
        ax.tick_params(labelsize=16)
        ax.legend(fontsize=14, loc='upper right')

    # Station and Depth as floats
    try:
        station_int = int(float(station))  
    except ValueError:
        station_int = station  

    try:
        depth_int = int(float(depth))  
    except ValueError:
        depth_int = depth

    plt.suptitle(f"Sample {sample_str}: \nStation {station_int} / Depth {depth_int} m", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save plot
    filename = os.path.join(
        output_dir,
        f"Vergleich_QSDWARF_AlleParameter_Station{station_int}_Depth{depth_int}_Sample_{sample_str}_{today_str}.pdf"
    )
    plt.savefig(filename, dpi=1200, format='pdf')
    plt.close()

    # save results per sample
    results.append({
        'Sample_idx': sample_idx,
        'Station': station,
        'Depth': depth,
        'Classical T_r': klassisch_values['T_r'][0],
        'Classical T_r err': klassisch_values['T_r'][1],
        'MC T_r': mc_fit['T_r'][0],
        'MC T_r std': mc_fit['T_r'][1],
        'Classical zeta': klassisch_values['zeta'][0],
        'Classical zeta err': klassisch_values['zeta'][1],
        'MC zeta': mc_fit['zeta'][0],
        'MC zeta std': mc_fit['zeta'][1],
        'Classical omega': klassisch_values['omega'][0],
        'Classical omega err': klassisch_values['omega'][1],
        'MC omega': mc_fit['omega'][0],
        'MC omega std': mc_fit['omega'][1],
        'S' : S
    })


#save results to a CSV file
output_file = os.path.join(output_dir, f"Comparison_Results_QSDWARF.csv")
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)


