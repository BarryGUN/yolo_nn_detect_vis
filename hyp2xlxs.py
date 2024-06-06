import os.path

import yaml
import pandas as pd


def yaml2xlsx(yml, project, name):
    # 读取YAML文件
    with open(yml, 'r') as file:
        data = yaml.safe_load(file)

    # 将YAML数据转换为DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')
    # df = df.transpose()
    if project != '' and not None:
        if not os.path.exists(project):
            os.makedirs(project)
        df.to_excel(f'{project}{os.sep}{name}', index=True)
    else:
        df.to_excel(f'{name}', index=True)

if __name__ == '__main__':
    yaml2xlsx(yml='data/hyps/hyp.scratch-vis.yaml',
              project='',
              name='hyp.xlsx')
