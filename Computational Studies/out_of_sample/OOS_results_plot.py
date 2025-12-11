import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

s = {
    1: 13428370.922167268, #['gamma_LT[CTV]: 1.0', 'gamma_LT[SOV]: 1.0']
    2: 13491989.55595761, #['gamma_ST[CTV,December]: 1.0', 'gamma_LT[CTV]: 1.0', 'gamma_LT[SOV]: 1.0']
    3: 13622001.677438546, #['gamma_ST[CTV,January]: 2.0', 'gamma_LT[CTV]: 1.0', 'gamma_LT[SOV]: 1.0']
    4: 13506080.043670759, #['gamma_ST[CTV,February]: 1.0', 'gamma_LT[CTV]: 1.0', 'gamma_LT[SOV]: 1.0']
    5: 13495176.398612758, #['gamma_ST[CTV,January]: 1.0', 'gamma_LT[CTV]: 1.0', 'gamma_LT[SOV]: 1.0']
    6: 13914180.940879101 #['gamma_LT[CTV]: 2.0', 'gamma_LT[SOV]: 1.0']
}
OOS_1 = [s[1], s[1], s[1], s[1], s[1], s[1], s[1], s[2], s[2], s[3]]
OOS_2 = [s[1], s[1], s[1], s[1], s[1], s[1], s[1], s[4], s[4], s[5]]
OOS_3 = []
OOS_4 = [s[1], s[1], s[1], s[1], s[1], s[1], s[1], s[1], s[1], s[4]]
OOS_5 = [s[1], s[1], s[1], s[1], s[1], s[1], s[1], s[1], s[1], s[1]]
OOS_6 = []
OOS_7 = []
OOS_8 = []
OOS_9 = [s[1], s[1], s[1], s[1], s[1], s[1], s[1], s[1], s[1], s[6]]
OOS_10 = [s[1], s[1], s[1], s[1], s[1], s[1], s[1], s[1], s[1], s[6]]

OOS_lists = [OOS_1, OOS_2, OOS_3, OOS_4, OOS_5, OOS_6, OOS_7, OOS_8, OOS_9, OOS_10]

means = []
cvs = []
ci_low = []
ci_high = []

for lst in OOS_lists:
    if len(lst) == 0:
        means.append(np.nan)
        cvs.append(np.nan)
        ci_low.append(np.nan)
        ci_high.append(np.nan)
        continue

    arr = np.array(lst)
    n = len(arr)
    mean = arr.mean()
    sd = arr.std(ddof=1)

    # CV
    cvs.append(sd / mean)

    # 95% CI using t-distribution
    t_crit = t.ppf(0.975, df=n-1)
    se = sd / np.sqrt(n)
    ci_l = mean - t_crit * se
    ci_h = mean + t_crit * se

    means.append(mean)
    ci_low.append(ci_l)
    ci_high.append(ci_h)

x = np.arange(1, 11)

# --- Plot MEAN + 95% CI ---
plt.figure(figsize=(10, 5))
plt.plot(x, means, marker='o', label='Mean')
plt.fill_between(x, ci_low, ci_high, alpha=0.2, label='95% CI')
plt.title("Mean across OOS sets with 95% CI")
plt.xlabel("OOS index")
plt.ylabel("Mean")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot CV ---
plt.figure(figsize=(10, 5))
plt.plot(x, cvs, marker='o', linestyle='-', color='orange', label='CV')
plt.title("CV across OOS sets")
plt.xlabel("OOS index")
plt.ylabel("CV")
plt.grid(True)
plt.tight_layout()
plt.show()