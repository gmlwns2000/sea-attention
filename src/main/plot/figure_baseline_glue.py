import json
import matplotlib.pyplot as plt
import os

import numpy as np
plt.style.use('seaborn-bright')

COLORS = {
    'perlin': 'magenta',
    'performer': 'blue',
    'reformer': 'purple',
    'scatterbrain': 'gray',
    'sinkhorn': 'orange',
    'synthesizer': 'yellow',
}
METHOD_NAMES = {
    'perlin': 'Ours',
    'performer': 'Performer',
    'reformer': 'Reformer',
    'scatterbrain': 'ScatterBrain',
    'sinkhorn': 'Sinkhorn',
    'synthesizer': 'Synthesizer',
}
MARKERS = {
    'perlin': '*'
}
MARKER_SIZE = {
    'perlin': 40,
    'default': 20
}
NAMES = {
    'cola': 'CoLA',
    'mrpc': 'MRPC',
    'mnli': 'MNLI',
}

def import_jn_format(path):
    """
    return {
        "{method}[,nbf:{nbf}][,k:{k}][,w:{w}]" : {
            "method": method,
            "metric": metric,
        }
    }
    "metric/f1" has priority
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    ret = {}
    for k, v in data.items():
        method = k.split()[0]
        if 'metric' in v:
            metric = float(v['metric']) * 100
        elif 'metric/f1' in v:
            metric = float(v['metric/f1']) * 100
        elif 'metric/accuracy' in v:
            metric = float(v['metric/accuracy']) * 100
        else:
            raise Exception(v)
        
        if method == 'perlin':
            nbf = v['nbf']
            k = v['k']
            w = v['w']
            ret[f'perlin,nbf:{nbf},k:{k},w:{w}'] = {
                'metric': metric,
                'method': 'perlin'
            }
        elif method == 'performer':
            nbf = v['nbf']
            ret[f'performer,nbf:{nbf}'] = {
                'metric': metric,
                'method': 'performer'
            }
        else:
            k = v['k']
            ret[f'{method},k:{k}'] = {
                'metric': metric,
                'method': method
            }
    return ret

def import_benchmark(path):
    """
    return {
        "{method}[,nbf:{nbf}][,k:{k}][,w:{w}]" : {
            "method": method,
            "latency": latency,
            "mem": mem,
        }
    }
    """
    with open(path, 'r') as f:
        return json.load(f)

data_cola = import_jn_format('./plots/main/bert_cola_ablation.json')
data_mrpc = import_jn_format('./plots/main/bert_mrpc_ablation.json')
metrics = {
    'cola': data_cola,
    'mrpc': data_mrpc,
    'mnli': data_mrpc,
}
benchmarks = import_benchmark('./plots/main/benchmark_bert_ablation/data.json')

methods = [set(map(lambda x: x['method'], v.values())) for v in metrics.values()]
methods = list(set.intersection(*methods))
methods.pop(methods.index('perlin'))
methods.insert(0, 'perlin')

root = './plots/main/figure_baseline_glue'
os.makedirs(root, exist_ok=True)

def render_fig(ax, data, benchmark, benchmark_metric='latency', x_label='ms'):
    plot_data = []
    for method in methods:
        xs = []
        ys = []
        for k, v in data.items():
            if v['method'] == method:
                y = v['metric']
                assert k in benchmark, k
                x = benchmark[k][benchmark_metric]
                xs.append(x)
                ys.append(y)
        
        ax.scatter(xs, ys, label=METHOD_NAMES[method], color=COLORS[method], marker=MARKERS.get(method, 'o'), s=MARKER_SIZE.get(method, MARKER_SIZE['default']))
        plot_data.append([xs, ys])
    ax.grid(True)
    ax.set_xlabel(x_label)
    return plot_data

ncols = len(metrics.keys())
nrows = 2

fig, axs = plt.subplots(nrows, ncols)
fig.set_figwidth(3.5*ncols)
fig.set_figheight(2.5*nrows+1)
# fig.set_facecolor('black')
fig.suptitle('Comparison of Trade-off Between Computational Cost and Accuracy')

all_plot_data = []
for isubset, subset in enumerate(metrics.keys()):
    ax = axs[0, isubset]
    ax.set_title(f'Memory / {NAMES[subset]}', fontsize=10)
    plot_data_memory = render_fig(ax, metrics[subset], benchmarks, 'mem', 'MB')
    
    ax = axs[1, isubset]
    ax.set_title(f'Latency / {NAMES[subset]}', fontsize=10)
    plot_data_latency = render_fig(ax, metrics[subset], benchmarks, 'latency', 'ms')
    all_plot_data.append([plot_data_memory, plot_data_latency])

handles, labels = ax.get_legend_handles_labels()
fig.subplots_adjust(bottom=0.135, hspace=0.4)
fig.legend(handles, labels, loc='lower center', ncol=len(labels))
# fig.tight_layout()

plt.savefig(os.path.join(root, 'plot_baseline_glue.png'), bbox_inches='tight')
plt.savefig(os.path.join(root, 'plot_baseline_glue.pdf'), bbox_inches='tight')

plt.clf()
plt.figure(figsize=(4,4))
for imethod, method in enumerate(methods):
    data = [
        [
            j[imethod]
            for j in i # for memory and latency
        ]
        for i in all_plot_data # for each subset
    ]
    # print(imethod, method, data)
    data = np.array(data)
    print(data.shape)
    # scale latency x axis
    data[:,1,0,:] = data[:,1,0,:] * 10
    data = data.mean(0).mean(0)
    xs = data[0]
    ys = data[1]
    print(data)
    plt.scatter(xs, ys, s=MARKER_SIZE.get(method, MARKER_SIZE['default']), label=METHOD_NAMES[method], color=COLORS[method], marker=MARKERS.get(method, 'o'))

# plt.legend()
plt.grid()
plt.xlabel('Average 10*Lat.+Mem.')
plt.ylabel('Average Metric')
plt.savefig(os.path.join(root, 'plot_baseline_glue_all.png'), bbox_inches='tight')
plt.savefig(os.path.join(root, 'plot_baseline_glue_all.pdf'), bbox_inches='tight')