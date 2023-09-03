import json
import matplotlib.pyplot as plt
import os
plt.style.use('seaborn-bright')

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
    for method in methods:
        xs = []
        ys = []
        for k, v in metric.items():
            if v['method'] == method:
                y = v['metric']
                x = benchmark[k][benchmark_metric]
                xs.append(x)
                ys.append(y)
        ax.scatter(
            xs, 
            ys, 
            s=MARKER_SIZE.get(method, MARKER_SIZE['default']), 
            marker=MARKERS.get(method, 'o'), 
            color=COLORS.get(method, 'gray'),
            label=METHOD_NAMES[method]
        )
    ax.grid(True)
    ax.set_xlabel(x_label)
    ax.set_ylabel('PPL.')

root = './plots/main/figure_baseline_opt'
os.makedirs(root, exist_ok=True)

nrows = 1
ncols = 2
fig, axs = plt.subplots(nrows, ncols)
fig.set_figwidth(3.5*ncols)
fig.set_figheight(2.5*nrows)
fig.suptitle('Comparison of Trade-off Between Computational Cost and Accuracy on Wikitext2')

ax = axs[0]
ax.set_title(f'Memory', fontsize=10)
render_plot(ax, metrics, benchmarks, 'latency', 'ms')

ax = axs[1]
ax.set_title(f'Latency', fontsize=10)
render_plot(ax, metrics, benchmarks, 'mem', 'MB')

handles, labels = ax.get_legend_handles_labels()
# fig.subplots_adjust(bottom=0.135)
fig.legend(handles, labels, loc='lower center', ncol=len(labels))

plt.savefig(os.path.join(root, 'plot_baseline_opt.png'), bbox_inches='tight')
plt.savefig(os.path.join(root, 'plot_baseline_opt.pdf'), bbox_inches='tight')