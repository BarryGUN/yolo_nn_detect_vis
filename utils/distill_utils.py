import math


def cosine_annealing_gain_decay(init_gain, epochs, epoch):
    if epoch > epochs:
        return 0
    else:
        # return self.hyp['distill'] * ((1 - math.cos(epoch * math.pi / self.epochs))
        #                               / 2) * (0.1 - 1) + 1
        return init_gain * (((1 - math.cos(epoch * math.pi / epochs))
                                      / 2) * (0.1 - 1) + 1)