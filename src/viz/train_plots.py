import matplotlib.pyplot as plt
import numpy as np


def train_plots(args, x_ax, y_ax,
                l_test, l_mean, l_std,
                y_mean, y_std,
                loss_type, plot_file):
    
    l_mean = np.array(l_mean)
    l_std = np.array(l_std)
    y_mean = np.array(y_mean)
    y_std = np.array(y_std)

    fig = plt.figure(figsize=(5, 4))
    ax1 = fig.add_subplot(111)
    ax2 = plt.twinx()
    l1 = ax1.plot(x_ax, y_ax, 'b.-.', label='accuracy')
    l1_ = ax1.plot(x_ax, y_mean, 'b', label='acc mean')
    ax1.fill_between(x_ax, y_mean-y_std, y_mean+y_std,
                     alpha=0.3, facecolor = 'b')
    ax1.set_ylabel('accuracy')
    l2 = ax2.plot(x_ax, l_test, 'r.-.', label='{} loss'.format(loss_type))
    l2_ = ax2.plot(x_ax, l_mean, 'r', label='{} mean'.format(loss_type))
    ax2.fill_between(x_ax, l_mean-l_std, l_mean+l_std,
                     alpha=0.3, facecolor = 'r')
    ax2.set_ylabel('{} loss'.format(loss_type))
    ax2.set_xlabel('epochs')
    ls = l1+l1_+l2+l2_
    lab = [_.get_label() for _ in ls]
    ax1.legend(ls, lab, loc=7)
    ax1.grid()
    plt.xlim(left=0, right=args.epochs)

    plt.savefig(plot_file, bbox_inches='tight', dpi=300)
    print('Saved: ', plot_file)

    return plot_file
