import matplotlib.pyplot as plt
import seaborn
seaborn.set()


def vis2d(data):
    y = data[:, 0]
    x1 = data[:, 1]
    x2 = data[:, 2]

    plt.plot(x1[y == 0], x2[y == 0], 'o', label='Class 0', markersize=3, color='red')
    plt.plot(x1[y == 1], x2[y == 1], 'o', label='Class 1', markersize=3, color='green')
    plt.legend()

    plt.show()


def pdb_on_error():
    import sys

    def info(type, value, tb):
        if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
            sys.__excepthook__(type, value, tb)
        else:
            try:
                import ipdb as pdb
            except ImportError:
                import pdb
            import traceback
            # we are NOT in interactive mode, print the exception
            traceback.print_exception(type, value, tb)
            print
            #  then start the debugger in post-mortem mode.
            # pdb.pm() # deprecated
            pdb.post_mortem(tb) # more

    sys.excepthook = info
