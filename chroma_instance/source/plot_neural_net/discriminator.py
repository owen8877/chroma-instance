import sys

sys.path.append('../')
from plot_neural_net.pycore.tikzeng import *

H = 40
H2 = H * 0.65
H4 = H2 * 0.65
H8 = H4 * 0.65

# defined your arch
arch = [
    to_head('.'),
    to_cor(),
    to_begin(),

    to_input('full_original.jpg', name='inputoriginal', to='(0, 5, 0)'),
    to_input('full_predicted.jpg', name='inputpredicted', to='(0, -5, 0)'),

    to_Conv(name='cr1', y_label='H/2', z_label='W/2', n_filer=(64, ''), offset="(4,0,0)", to="(0,0,0)",
                    width=(2,), height=H, depth=H, caption="conv1"),
    to_Conv(name='cr2', y_label='H/4', z_label='W/4', n_filer=(128, ''), offset="(3,0,0)", to="(cr1-east)",
                    width=(4,), height=H2, depth=H2, caption="conv2"),
    to_Conv(name='cr3', y_label='H/8', z_label='W/8', n_filer=(256, ''), offset="(3,0,0)", to="(cr2-east)",
                    width=(8,), height=H4, depth=H4, caption="conv3"),
    to_Conv(name='cr4', y_label='H/8', z_label='W/8', n_filer=(512, ''), offset="(3,0,0)", to="(cr3-east)",
                    width=(16,), height=H4, depth=H4, caption="conv4"),
    to_Conv(name='cr5', y_label='H/8', z_label='W/8', n_filer=(1, ''), offset="(3,0,0)", to="(cr4-east)",
                    width=(1,), height=H4, depth=H4, caption="conv5"),


    # connections
    to_connection("cr1", "cr2"),
    to_connection("cr2", "cr3"),
    to_connection("cr3", "cr4"),
    to_connection("cr4", "cr5"),

    to_end(),
]


def main():
    # namefile = str(sys.argv[0]).split('.')[0]
    # to_generate(arch, f'../../../figs/{namefile}.tex')
    for c in arch:
        print(c)


if __name__ == '__main__':
    main()
