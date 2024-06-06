import os

import matplotlib.pyplot as plt
import pandas as pd

class Model:
    def __init__(self, latency, ap, name, color, mark):
        self.latency = latency
        self.ap = ap
        self.name = name
        self.color = color
        self.mark = mark

    def __call__(self, *args, **kwargs):
        dict_items = {}
        for attr in dir(self):
            if not attr.startswith("__") and not callable(getattr(self, attr)):
                dict_items[attr] = getattr(self, attr)

        return dict_items

def get_n_colors(n):
    colors = plt.cm.get_cmap('hsv', n)  # 这里使用了 'hsv' 调色板
    return [colors(i) for i in range(n)]


def get_data(xls_path_bs1, xls_path_bs35):
    df_bs1 = pd.read_excel(xls_path_bs1, sheet_name='Sheet1')
    df_bs35 = pd.read_excel(xls_path_bs35, sheet_name='Sheet1')
    assert len(df_bs1['model'].unique()) == len(df_bs35['model'].unique())
    assert set(df_bs1['model'].unique()) == set(df_bs35['model'].unique())
    names = df_bs1['model'].unique()
    models_bs1 = []
    models_bs35 = []
    colors = get_n_colors(len(names))
    markers = get_marker(len(names))
    for i, name in enumerate(names):
        latency = []
        ap = []
        for index, row in df_bs35.iterrows():
            if row['model'] == name:
                latency.append(row['latency'])
                ap.append(row['AP'])
        models_bs35.append(Model(
            name=f'{name}(ours)' if 'yolonn' in name else name,
            latency=latency,
            ap=ap,
            color=colors[i],
            mark=markers[i]
        ))

    for i, name in enumerate(names):
        latency = []
        ap = []
        for index, row in df_bs1.iterrows():
            if row['model'] == name:
                latency.append(row['latency'])
                ap.append(row['AP'])
        models_bs1.append(Model(
            name=f'{name}(ours)' if 'yolonn' in name else name,
            latency=latency,
            ap=ap,
            color=colors[i],
            mark=markers[i]
        ))
    return models_bs35, models_bs1

def get_marker(n):
    markers = ['o', '.', 'x', '+', '*', 's', 'D', 'd',
               'p', 'h', 'H', 'v', '^', '<', '>', '1', '2',
               '3', '4', '|', '_']

    return [markers[i] for i in range(n)]


def speed_plot(speed_models, title, project='', name=''):
    # x = [1, 2, 3, 4, 5]
    # y = [2, 3, 5, 7, 11]
    for item in speed_models:
        model_dict = item()
        plt.plot(model_dict['latency'],
                 model_dict['ap'],
                 marker=model_dict['mark'],
                 color=model_dict['color'],
                 label=model_dict['name'])
    plt.title(title)
    plt.xlabel("Latency")
    plt.ylabel("AP")
    plt.legend()
    plt.grid(True)

    if project != '' and project is not None:
        if not os.path.exists(project):
            os.makedirs(project)
        plt.savefig(f"{project}{os.sep}{name}")
    else:
        plt.savefig(f"{name}")

    plt.show()


if __name__ == '__main__':

    speed_models_bs35, speed_models_bs1 = get_data(xls_path_bs35='speed/tensorrt-bs35.xlsx',
                                                   xls_path_bs1='speed/tensorrt-bs1.xlsx')
    speed_plot(speed_models=speed_models_bs35,
               title='latency',
               name='speed/bs35.jpg')

    speed_plot(speed_models=speed_models_bs1,
               title='latency',
               name='speed/bs1.jpg')