import numpy as np
from matplotlib import pyplot as plt

# 8
# 0.095
# 0.05
# 0.075
# 2
def align_new(x, y, alpha=0.5, beta=0.095):
    return pow(x, alpha) * abs((beta * y / ((beta + 1) - y)))

# # beta=0.095
# # beta =1
# def align_org(x, y, alpha=1.0, beta=0.095):
#     return pow(x, alpha) * pow(y, beta)

# def align_new(x, y, alpha=1.0, beta=0.095):
#     return pow(x, alpha) * (beta * y / ((beta + 1) - y))
#
def align_org(x, y, alpha=0.5, beta=6.0):
    return pow(x, alpha) * pow(y, beta)


def scatter_plot(npz, fcns, topk):
    loaded_data = np.load(npz)
    xys = (loaded_data['xs'], loaded_data['ys'])
    matrics = {}
    for fcn in fcns:
        results = []
        for i, (x, y) in enumerate(zip(xys[0], xys[1])):
            results.append((i, x, y, fcn(x, y)))
        matrics[fcn.__name__] = sorted(results, key=lambda v: v[-1], reverse=True)[:topk]


    for key in matrics.keys():
        plt.figure()
        plt.scatter(xys[0], xys[1], c='blue', s=2)
        xk = [e[1] for e in matrics[key]]
        yk = [e[2] for e in matrics[key]]
        plt.scatter(xk, yk, c='red', label=key, s=2)
        plt.legend(loc='upper left')
        plt.title('align visual')
        plt.xlabel('bbox-score')
        plt.ylabel('overlaps')
        plt.show()




if __name__ == '__main__':
    # mean = 0
    # std_dev = 1
    #
    # np.savez('data_3.npz', xs=np.random.rand(8400), ys=np.random.rand(8400))
    fcns = [align_new, align_org]
    # # # fcns = [ align_org]
    scatter_plot('data_2.npz',fcns=fcns, topk=10)