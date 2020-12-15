import sys

sys.path.append('../')
from plot_neural_net.pycore.tikzeng import *

H = 40
H2 = H * 0.65
H4 = H2 * 0.65
H8 = H4 * 0.65
H16 = H8 * 0.65
H32 = H16 * 0.65
H0 = 1

T1 = 1
T2 = 1
T32 = 1.5
T64 = 2
T128 = 2.75
T256 = 3.5
T512 = 6

# defined your arch
arch = [
    to_head('.'),
    to_cor(),
    to_begin(),

    to_Conv(name='inputdummy', y_label='H', z_label='W', n_filer=('', ''), width=(1,), height=H, depth=H,
            fill_color=r"\BlankColor"),
    to_input('full_original_gray.jpg', to="(inputdummy-east)"),

    to_Conv(name='cr1', y_label='H', z_label='W', n_filer=(64, 64), offset="(4,0,0)", to="(0,0,0)",
            width=(T64, T64), height=H, depth=H),
    to_Conv(name='cr2', y_label='H/2', z_label='W/2', n_filer=(128, 128), offset="(2,0,0)", to="(cr1-east)",
            width=(T128, T128), height=H2, depth=H2),
    to_Conv(name='cr3', y_label='H/4', z_label='W/4', n_filer=(256, 256, 256), offset="(2,0,0)",
            to="(cr2-east)", width=(T256, T256, T256), height=H4, depth=H4),
    to_Conv(name='cr4', y_label='H/8', z_label='W/8', n_filer=(512, 512, 512), offset="(2,0,0)",
            to="(cr3-east)", width=(T512, T512, T512), height=H8, depth=H8),
    to_Conv(name='cr4d', y_label='', z_label='', n_filer=('',''), offset="(4.4,0,0)", fill_color=r"\BlankColor",
            to="(cr3-east)", width=(T512,), height=H8, depth=H8),
    to_inj(name='inj1', offset="(0,1,0)", to="(cr4d-north)"),

    to_Conv(name='cr5', y_label='H/16', z_label='W/16', n_filer=(512, 512), offset="(1,0,15)", to="(cr4-south)",
            width=(T512, T512), height=H16, depth=H16, fill_color=r"\RedColor"),
    to_Conv(name='cr6', y_label='H/32', z_label='W/32', n_filer=(512, ''), offset="(2,0,0)", to="(cr5-east)",
            width=(T512,), height=H32, depth=H32, fill_color=r"\RedColor"),
    to_Conv(name='cr7', y_label='H/32', z_label='W/32', n_filer=(512, ''), offset="(2.5,0,0)", to="(cr6-east)",
            width=(T512,), height=H32, depth=H32, fill_color=r"\RedColor"),

    to_FullyConnected(name='f1', s_filer=4096, n_filer=1, offset="(5,0,0)", to="(cr7-east)", width=T1, depth=H,
                      height=H0),
    to_FullyConnected(name='f2', s_filer=4096, n_filer=1, offset="(1.5,0,0)", to="(f1-east)", width=T1, depth=H,
                      height=H0),
    to_FullyConnected(name='f3', s_filer=1000, n_filer=1, offset="(1.5,0,0)", to="(f2-east)", width=T1, depth=H4,
                      height=H0),

    to_Conv(name='fla1', y_label='', z_label='1', n_filer=(1024, ''), offset="(-1,0,-3)", to="(cr7-north)",
            width=(T512 * 2.5,), height=H0, depth=H0, fill_color=r"\RedColor"),
    to_Conv(name='fla2', y_label='', z_label='1', n_filer=(512, ''), offset="(0,0,-2)", to="(fla1-north)",
            width=(T512,), height=H0, depth=H0, fill_color=r"\RedColor"),
    to_Conv(name='fla3', y_label='', z_label='1', n_filer=(256, ''), offset="(0,0,-2)", to="(fla2-north)",
            width=(T256,), height=H0, depth=H0, fill_color=r"\RedColor"),

    to_Conv(name='mcr8', y_label='H/8', z_label='W/8', n_filer=(512, ''), offset="(2.5,0,0)",
            to="(cr4-east)", width=(T512,), height=H8, depth=H8, fill_color=r"\GreyColor"),
    to_Conv(name='mcr9', y_label='H/8', z_label='', n_filer=(256, ''), offset="(2.5,0,0)",
            to="(mcr8-east)", width=(T512 / 2,), height=H8, depth=H8, fill_color=r"\GreyColor"),
    to_inj(name='inj2', offset="(0,1,0)", to="(mcr9-north)"),
    to_Conv(name='mcr10dummy', y_label='', z_label='W/8', n_filer=(256, ''), offset="(0,0,0)",
            to="(mcr9-east)", width=(T512 / 2,), height=H8, depth=H8, fill_color=r"\RedColor"),
    to_Conv(name='fla3mcr10', y_label='', z_label='', n_filer=('', ''), offset="(0,0,0)", to="(mcr9-east)",
            width=(T256,), height=H0, depth=H0, fill_color=r"\RedColor"),

    to_Conv(name='cr11', y_label='H/8', z_label='W/8', n_filer=(256, 128), offset="(2,0,0)", to="(fla3mcr10-east)",
            width=(T256, T128), height=H8, depth=H8, fill_color=r"\BlueColor"),
    to_Conv(name='cr11d', y_label='', z_label='', n_filer=('', ''), width=(T128,), height=H8, depth=H8,
            fill_color=r"\BlankColor", offset=f"(2.7,0,0)", to="(fla3mcr10-east)"),
    to_inj(name='inj3', offset="(0,1,0)", to="(cr11d-north)"),

    to_Conv(name='cr12', y_label='H/4', z_label='W/4', n_filer=(64, 64), offset="(2.5,0,0)", to="(cr11-east)",
            width=(T64, T64), height=H4, depth=H4, fill_color=r"\BlueColor"),
    to_Conv(name='cr12d', y_label='', z_label='', n_filer=('', ''), width=(T64,), height=H4, depth=H4,
            fill_color=r"\BlankColor", offset=f"(2.9,0,0)", to="(cr11-east)"),
    to_inj(name='inj4', offset="(0,1,0)", to="(cr12d-north)"),

    to_Conv(name='cr13', y_label='H/2', z_label='W/2', n_filer=('32 2', 2), offset="(3,0,0)", to="(cr12-east)",
            width=(T32, T2), height=H2, depth=H2, fill_color=r"\BlueColor"),
    to_Conv(name='cr13d', y_label='', z_label='', n_filer=('', ''), width=(T2,), height=H2, depth=H2,
            fill_color=r"\BlankColor", offset=f"(3.3,0,0)", to="(cr12-east)"),
    to_inj(name='inj5', offset="(0,1,0)", to="(cr13d-north)"),

    to_Conv(name='cr14', y_label='H', z_label='W', n_filer=('2', ''), width=(1,), height=H, depth=H,
            fill_color=r"\BlankColor", offset="(3,0,0)", to="(cr13-east)"),
    to_input('full_intermediate.jpg', to="(cr14-east)"),

    # connections
    to_connection("cr1", "cr2"),
    to_connection("cr2", "cr3"),
    to_connection("cr3", "cr4"),

    to_connection("cr4", "cr5"),
    to_connection("cr5", "cr6"),
    to_connection("cr6", "cr7"),

    to_connection("cr7", "f1"),
    to_connection("f1", "f2"),
    to_connection("f2", "f3"),

    to_connection_thin("cr7", "fla1", of_dir='west', to_dir='west'),
    to_connection_thin("cr7", "fla1", of_dir='east', to_dir='east'),
    to_connection_thin("fla1", "fla2", of_dir='west', to_dir='west'),
    to_connection_thin("fla1", "fla2", of_dir='east', to_dir='east'),
    to_connection_thin("fla2", "fla3", of_dir='west', to_dir='west'),
    to_connection_thin("fla2", "fla3", of_dir='east', to_dir='east'),
    to_connection_thin("fla3", "fla3mcr10", of_dir='west', to_dir='west'),
    to_connection_thin("fla3", "fla3mcr10", of_dir='east', to_dir='east'),

    to_connection("cr4", "mcr8"),
    to_connection("mcr8", "mcr9"),

    to_connection("mcr10dummy", "cr11"),
    to_connection("cr11", "cr12"),
    to_connection("cr12", "cr13"),
    to_connection("cr13", "cr14"),

    to_connection("cr4d", "inj1", of_dir='north', to_dir='south'),
    to_connection("mcr9", "inj2", of_dir='north', to_dir='south'),
    to_connection("cr11d", "inj3", of_dir='north', to_dir='south'),
    to_connection("cr12d", "inj4", of_dir='north', to_dir='south'),
    to_connection("cr13d", "inj5", of_dir='north', to_dir='south'),

    # to_upsample('cr11', 'cr12'),
    # to_upsample('cr12', 'cr13'),
    # to_upsample('cr13', 'cr14'),

    to_end(),
]


def main():
    # namefile = str(sys.argv[0]).split('.')[0]
    # to_generate(arch, f'../../../figs/{namefile}.tex')
    for c in arch:
        print(c)


if __name__ == '__main__':
    main()
