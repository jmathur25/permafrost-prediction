# %%
import sys
from matplotlib import pyplot as plt
import pandas as pd

sys.path.append("..")

# %%
from download_temp import download_year

df_2003 = download_year(2003)

# %%
df_2003

# %%
df = pd.read_excel("U1_clim.xls", sheet_name="data", nrows=366, parse_dates=["DATE"])

# %%
df.shape, df_2003.shape, df_2004.shape
# %%
plt.plot(df[2004], color="r")
plt.plot(df_2004["temp_2m_c"], color="b")


# %%
def calc_bias(d1, d2):
    indices = d1 > 0
    return (d1[indices] - d2[indices]).mean()


bias = calc_bias(df[2004], df_2004["temp_2m_c"])
bias

# %%
plt.plot(df[2003], color="r")
plt.plot(df_2003["temp_2m_c"], color="b")
