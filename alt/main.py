import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

signals = pd.read_csv("signals.csv")
model = pd.read_csv("model.csv")
summary = pd.read_csv("summary.csv")

gamma = summary.loc[0, "gamma"]
misfit = summary.loc[0, "misfit"]

print(f"Gamma = {gamma}")
print(f"Misfit = {misfit}")

# 1. График наблюденного и рассчитанного поля
plt.figure(figsize=(10, 4))
plt.plot(signals["x"], signals["g_obs"], label="Практические данные")
plt.plot(signals["x"], signals["g_calc"], "--", label="Расчетные данные")
plt.title("Сравнение практических и расчетных значений Δg")
plt.xlabel("x")
plt.ylabel("Δg")
plt.grid(True)
plt.legend()
plt.tight_layout()

# 2. График невязки
plt.figure(figsize=(10, 4))
plt.plot(signals["x"], signals["residual"])
plt.title("Отклонение практических и расчетных данных")
plt.xlabel("x")
plt.ylabel("g_obs - g_calc")
plt.grid(True)
plt.tight_layout()

# 3. Карта истинной модели
nx = model["ix"].max() + 1
nz = model["iz"].max() + 1

true_grid = np.full((nz, nx), np.nan)
est_grid = np.full((nz, nx), np.nan)

for _, row in model.iterrows():
    ix = int(row["ix"])
    iz = int(row["iz"])
    true_grid[iz, ix] = row["rho_true"]
    est_grid[iz, ix] = row["rho_est"]

plt.figure(figsize=(8, 4))
plt.imshow(true_grid, origin="lower", aspect="auto")
plt.colorbar(label="Плотность")
plt.title("Истинное распределение плотности")
plt.xlabel("ix")
plt.ylabel("iz")
plt.tight_layout()

# 4. Карта восстановленной модели
plt.figure(figsize=(8, 4))
plt.imshow(est_grid, origin="lower", aspect="auto")
plt.colorbar(label="Плотность")
plt.title("Восстановленное распределение плотности")
plt.xlabel("ix")
plt.ylabel("iz")
plt.tight_layout()

plt.show()