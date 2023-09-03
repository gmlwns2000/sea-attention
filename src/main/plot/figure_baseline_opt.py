import json
import matplotlib.pyplot as plt
import os
plt.style.use('seaborn-bright')

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
NAMES = {
    'cola': 'CoLA',
    'mrpc': 'MRPC',
    'mnli': 'MNLI',
}

def load_metrics(path):
    pass

def load_benchmark(path):
    pass

metrics = load_metrics('./plots/main/opt_albation.json')
benchmarks = load_benchmark('./plots/main/benchmark_opt_ablation/data.json')

