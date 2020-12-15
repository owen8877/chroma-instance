import sys

sys.path.append('../')
from plot_neural_net.pycore.tikzeng import *

H = 15
H2 = H * 0.65
H4 = H2 * 0.65
H8 = H4 * 0.65

# defined your arch
arch = [
    to_head('.'),
    to_cor(),
    to_begin(),

    to_Conv(name='cr1', y_label='H', z_label='W', n_filer=('u', ''), offset="(0,0,0)", to="(0,0,0)",
                    width=(4,), height=H, depth=H),
    to_Conv(name='cr2', y_label='H', z_label='W', n_filer=('u', ''), offset="(4.5,0,0)", to="(cr1-east)",
                    width=(4,), height=H, depth=H),
    to_Conv(name='cr3', y_label='H', z_label='W', n_filer=(1, ''), offset="(4.5,0,0)", to="(cr2-east)",
                    width=(1,), height=H, depth=H),


    # connections
    to_connection("cr1", "cr2"),
    to_connection("cr2", "cr3"),

    to_end(),
]


def main():
    # namefile = str(sys.argv[0]).split('.')[0]
    # to_generate(arch, f'../../../figs/{namefile}.tex')
    for c in arch:
        print(c)


if __name__ == '__main__':
    main()
