import math
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-bright')
matplotlib.rcParams['font.family'] = 'Noto Sans, DejaVu Sans'

data = pd.read_csv("./plots/main/wandb_opt125_openwebtext.csv")
colnames = [
    'opt125 k64 w64 openwebtext - eval/score',
    'opt125 performer openwebtext - eval/score',
]
names = [
    'Ours',
    'Performer',
]
TOMETHOD = {
    'Ours':'perlin',
    'Performer':'performer',
}
COLORS = {
    'none': 'green',
    'perlin': 'magenta',
    'performer': 'blue',
    'reformer': 'purple',
    'scatterbrain': 'gray',
    'sinkhorn': 'orange',
    'synthesizer': 'yellow',
}

xss = []
yss = []

for cn in colnames:
    dxs = data['Step'].to_numpy()
    dys = data[cn].to_numpy()
    xs = []
    ys = []
    for i in range(len(dys)):
        x = dxs[i]
        y = dys[i]
        if not math.isnan(y):
            xs.append(x)
            ys.append(y)
    xss.append(xs)
    yss.append(ys)

plt.figure(figsize=(3.5,2.7))

for i in range(len(xss)):
    name = names[i]
    xs = xss[i]
    ys = yss[i]
    plt.plot(xs, ys, label=name, color=COLORS[TOMETHOD[name]])

plt.grid()
plt.legend()
plt.ylim(0, 150)
plt.xlabel('Optimizer Steps', fontweight=500)
plt.ylabel('PPL. ↓', fontweight=500)
plt.title('Validation Curve', fontweight=500)

root = './plots/main/figure_opt_curve_openwebtext'
os.makedirs(root, exist_ok=True)
plt.savefig(os.path.join(root, 'plot_opt_curve_openwebtext.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(root, 'plot_opt_curve_openwebtext.png'), bbox_inches='tight')
print(os.path.join(root, 'plot_opt_curve_openwebtext.png'))