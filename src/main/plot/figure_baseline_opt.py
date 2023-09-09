import json
import matplotlib.pyplot as plt
import os

import numpy as np
plt.style.use('seaborn-bright')
import matplotlib

matplotlib.rcParams['font.family'] = 'Noto Sans'

METHOD_NAMES = {
    'perlin': 'Ours',
    'performer': 'Performer',
    'reformer': 'Reformer',
    'scatterbrain': 'ScatterBrain',
    'sinkhorn': 'Sinkhorn',
    'synthesizer': 'Synthesizer',
}
COLORS = {
    'perlin': 'magenta',
    'performer': 'blue',
    'reformer': 'purple',
    'scatterbrain': 'gray',
    'sinkhorn': 'orange',
    'synthesizer': 'yellow',
}
MARKERS = {
    'perlin': '*'
}
MARKER_SIZE = {
    'perlin': 40,
    'default': 20
}

def load_metrics(path):
    with open(path, 'r') as f:
        data = json.load(f)
    for k, v in data.items():
        v['method'] = k.split(',')[0]
    return data

def load_benchmark(path):
    with open(path, 'r') as f:
        return json.load(f)

metrics = load_metrics('./plots/main/opt_albation.json')
benchmarks = load_benchmark('./plots/main/benchmark_opt_ablation/data.json')
methods = ['perlin', 'reformer', 'performer']

def render_plot(ax, metric, benchmark, benchmark_metric, x_label):
    plot_data = []
    for method in methods:
        xs = []
        ys = []
        for k, v in metric.items():
            if v['method'] == method:
                y = v['metric']
                x = benchmark[k][benchmark_metric]
                xs.append(x)
                ys.append(y)
        # print(xs, ys, method)
        ax.scatter(
            xs, 
            ys, 
            s=MARKER_SIZE.get(method, MARKER_SIZE['default']), 
            marker=MARKERS.get(method, 'o'), 
            color=COLORS.get(method, 'gray'),
            label=METHOD_NAMES[method]
        )
        plot_data.append([xs, ys])
    ax.grid(True)
    ax.set_xlabel(x_label)
    ax.set_ylabel('PPL. (Lower is better)')
    return plot_data

root = './plots/main/figure_baseline_opt'
os.makedirs(root, exist_ok=True)

nrows = 1
ncols = 3
fig, axs = plt.subplots(nrows, ncols)
fig.set_figwidth(3.5*ncols)
fig.set_figheight(3*nrows)
fig.suptitle('Comparison of Trade-off Between Computational Cost and Accuracy on Wikitext2', fontsize=12, fontweight=500)

ax = axs[1]
ax.set_title(f'Memory', fontsize=11, fontweight=500)
plot_data_mem = render_plot(ax, metrics, benchmarks, 'mem', 'MB')

ax = axs[2]
ax.set_title(f'Latency', fontsize=11, fontweight=500)
plot_data_latency = render_plot(ax, metrics, benchmarks, 'latency', 'ms')

ax = axs[0]
plot_data = [plot_data_mem, plot_data_latency]
ax.set_title(f'Overall Efficiency', fontsize=11, fontweight=500)
for imethod, method in enumerate(methods):
    data = [
        i[imethod]
        for i in plot_data 
    ]
    data = np.array(data)
    data[1,0,:] = data[1,0,:]*10
    data = data.mean(0)
    # print(method, data)
    xs = data[0, :]
    ys = data[1, :]
    ax.scatter(
        xs, 
        ys, 
        s=MARKER_SIZE.get(method, MARKER_SIZE['default']), 
        marker=MARKERS.get(method, 'o'), 
        color=COLORS.get(method, 'gray'),
        label=METHOD_NAMES[method]
    )
    ax.grid(True)
    ax.set_xlabel('10*Lat.+Mem.', fontweight=500)
    ax.set_ylabel('PPL. (Lower is better)', fontweight=500)

handles, labels = ax.get_legend_handles_labels()
fig.subplots_adjust(top=0.82, bottom=0.27, wspace=0.3)
fig.legend(handles, labels, loc='lower center', ncol=len(labels))

plt.savefig(os.path.join(root, 'plot_baseline_opt.png'), bbox_inches='tight')
plt.savefig(os.path.join(root, 'plot_baseline_opt.pdf'), bbox_inches='tight')
print(os.path.join(root, 'plot_baseline_opt.png'))