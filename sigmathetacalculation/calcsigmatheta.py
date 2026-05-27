import pandas as pd
import numpy as np
import gsw

bottledata = pd.read_csv(
    "sigmathetacalculation/SO21_bottle.tab", skiprows=160, sep="\t"
)
ngdata = pd.read_csv(
    "sigmathetacalculation/Noble_Gas_Data_SAS21_VACAO.tab", skiprows=53, sep="\t"
)

df = pd.merge(bottledata, ngdata)
print(len(df))
df = df.rename(columns={"Depth water [m]": "Depth"})

sigmathetaseries = gsw.density.sigma0(df["ASAL [g/kg]"], df["Tcon [°C]"])
df["sigma0 [kg/m**3]"] = sigmathetaseries
df.to_csv("sigmathetacalculation/VACAO_Data_With_SigmaTheta")
