import sys
import argparse

from . import sk_bias

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='command')

make_interpolation = subparsers.add_parser('make_interpolation')
make_interpolation.add_argument('pkl_outfile')

check_interpolation = subparsers.add_parser('check_interpolation')
check_interpolation.add_argument('pkl_infile')

make_plot = subparsers.add_parser('make_plot')
make_plot.add_argument('pkl_infile')
make_plot.add_argument('pdf_outfile')

test = subparsers.add_parser('test')

args = parser.parse_args()

if args.command == 'make_interpolation':
    interp = sk_bias.BiasInterpolator()
    sk_bias.write_pickle(args.pkl_outfile, interp)
elif args.command == 'check_interpolation':
    interp = sk_bias.read_pickle(args.pkl_infile)
    interp.check_interpolation()
elif args.command == 'make_plot':
    interp = sk_bias.read_pickle(args.pkl_infile)
    interp.make_plot(args.pdf_outfile)
elif args.command == 'test':
    sk_bias.test_ipow()
    sk_bias.MCTracker.test()
else:
    parser.print_help()
    sys.exit(2)

