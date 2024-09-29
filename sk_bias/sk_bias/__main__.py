import sys
import argparse

from . import sk_bias


def do_run_mcs2(rms, n, mu_min=0.0, mu_max=98.0):
    """No bvec. Runs forever!"""
    
    pdf = sk_bias.Pdf(rms)
    min_s1 = max(round(mu_min*n), 1)
    max_s1 = min(round(mu_max*n), 98*n)
    print(f'do_run_mcs2: {rms=} {n=} {mu_min=} {mu_max=} {min_s1=} {max_s1=}')
    pdf.run_mcs(n, min_s1=min_s1, max_s1=max_s1)


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

run_mcs2 = subparsers.add_parser('run_mcs2')
run_mcs2.add_argument('rms', type=float)
run_mcs2.add_argument('n', type=int)
run_mcs2.add_argument('mu_min', type=float, nargs='?', default=0.0)
run_mcs2.add_argument('mu_max', type=float, nargs='?', default=98.0)

run_unquantized_mcs = subparsers.add_parser('run_unquantized_mcs')
run_unquantized_mcs.add_argument('n', type=int)

run_transit_mcs = subparsers.add_parser('run_transit_mcs')
run_transit_mcs.add_argument('nt', type=int)
run_transit_mcs.add_argument('ndish', type=int)
run_transit_mcs.add_argument('brightness', type=float)

emit_code = subparsers.add_parser('emit_code')
emit_code.add_argument('pkl_infile')

test = subparsers.add_parser('test')

args = parser.parse_args()

if args.command == 'make_interpolation':
    interp = sk_bias.BiasInterpolator()
    sk_bias.write_pickle(args.pkl_outfile, interp)
elif args.command == 'check_interpolation':
    interp = sk_bias.read_pickle(args.pkl_infile)
    interp.check_bias_interpolation()
    interp.check_sigma_interpolation()
elif args.command == 'make_plot':
    interp = sk_bias.read_pickle(args.pkl_infile)
    interp.make_plot(args.pdf_outfile)
elif args.command == 'run_mcs':
    interp = sk_bias.read_pickle(args.pkl_infile)
    interp.run_mcs(args.rms, args.n, verbose = args.verbose)
elif args.command == 'run_mcs2':
    do_run_mcs2(args.rms, args.n, args.mu_min, args.mu_max)
elif args.command == 'run_unquantized_mcs':
    sk_bias.run_unquantized_mcs(args.n)
elif args.command == 'run_transit_mcs':
    sk_bias.run_transit_mcs(args.nt, args.ndish, args.brightness)
elif args.command == 'emit_code':
    interp = sk_bias.read_pickle(args.pkl_infile)
    sk_bias.emit_code(interp)
elif args.command == 'test':
    sk_bias.test_ipow()
    sk_bias.MCTracker.test()
    sk_bias.test_fit_polynomial()
else:
    parser.print_help()
    sys.exit(2)

