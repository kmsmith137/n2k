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

run_mcs = subparsers.add_parser('run_mcs')
run_mcs.add_argument('pkl_infile')
run_mcs.add_argument('rms', type=float)
run_mcs.add_argument('n', type=int)
run_mcs.add_argument('-v', '--verbose', action='store_true')

emit_code = subparsers.add_parser('emit_code')
emit_code.add_argument('pkl_infile')

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
elif args.command == 'run_mcs':
    interp = sk_bias.read_pickle(args.pkl_infile)
    interp.run_mcs(args.rms, args.n, verbose = args.verbose)
elif args.command == 'emit_code':
    interp = sk_bias.read_pickle(args.pkl_infile)
    interp.emit_code()
elif args.command == 'test':
    sk_bias.test_ipow()
    sk_bias.MCTracker.test()
    sk_bias.test_fit_polynomial()
else:
    parser.print_help()
    sys.exit(2)

