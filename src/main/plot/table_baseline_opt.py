data = {
    'opt-125m': {
        'none': 22.0,
        'ours': 28.0,
        'reformer': 97.0,
        'performer': 132.0,
    },
    'opt-350m': {
        'none': 20.0,
        'ours': 20.7,
        'reformer': 70.0,
        'performer': 0.0,
    },
    'opt-1.3b': {
        'none': 13.0,
        'ours': 0.0,
        'reformer': 0.0,
        'performer': 0.0,
    },
    'opt-2.7b': {
        'none': 12.0,
        'ours': 0.0,
        'reformer': 0.0,
        'performer': 0.0,
    },
    'opt-6.7b': {
        'none': 10.0,
        'ours': 0.0,
        'reformer': 0.0,
        'performer': 0.0,
    }
}

aliases = {
    'opt-125m': '125M',
    'opt-350m': '350M',
    'opt-1.3b': '1.3B',
    'opt-2.7b': '2.7B',
    'opt-6.7b': '6.7B',
    'none': 'None',
    'ours': 'Ours (k=64)',
    'reformer': 'Reformer',
    'performer': 'Performer',
}
col_keys = ['opt-125m', 'opt-350m', 'opt-1.3b', 'opt-2.7b', 'opt-6.7b']
row_keys = ['none', 'ours', 'reformer', 'performer']

cell_data = []
for row_name in row_keys:
    cell_data.append([row_name] + [data[c][row_name] for c in col_keys])
just_width = 13
just_width_header = 10
def format(lst):
    return [f'{i:.1f}'.rjust(just_width) for i in lst]
table_data = ""
for row in cell_data:
    if isinstance(row, list):
        name, row_data = row[0], row[1:]
        table_data += '&'.join([aliases[name].rjust(just_width_header)]+format(row_data))
        table_data += '\\\\\n'
    elif isinstance(row, str):
        table_data += row
    else:
        raise Exception()

table_cells = "|".join(['c'] * len(cell_data[0]))
table_header = "&".join(["".rjust(just_width_header)] + [aliases[k].rjust(just_width) for k in col_keys])
table = \
f"\\begin{{table}}[h]\n"\
f"\\caption{{opt benchmark}}\n"\
f"\\label{{table.baseline.opt}}\n"\
f"\\begin{{center}}\n"\
f"\\begin{{tabular}}{{{table_cells}}}\n"\
f"{table_header}\\\\\n"\
f"\\hline\n"\
f"{table_data}"\
f"\\end{{tabular}}\n"\
f"\\end{{center}}\n"\
f"\\end{{table}}"

print(table)