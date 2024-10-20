import sys
import argparse

from . import sk_bias


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='command')

test = subparsers.add_parser('test')

check_interpolation = subparsers.add_parser('check_interpolation')

run_mcs = subparsers.add_parser('run_mcs')
run_mcs.add_argument('rms', type=float)
run_mcs.add_argument('n', type=int)

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

args = parser.parse_args()

if args.command == 'test':
    sk_bias.test_ipow()
    sk_bias.test_fit_polynomial()
    sk_bias.test_mc_tracker()
elif args.command == 'check_interpolation':
    binterp = sk_bias.BiasInterpolator()
    sk_bias.check_bias_interpolation(binterp)
    sk_bias.check_sigma_interpolation(binterp)
elif args.command == 'run_mcs':
    pdf = sk_bias.Pdf(rms = args.rms)
    binterp = sk_bias.BiasInterpolator()
    sk_bias.run_mcs(pdf, args.n, binterp = binterp)
elif args.command == 'run_mcs2':
    pdf = sk_bias.Pdf(args.rms)
    min_s1 = max(round(args.mu_min * args.n), 1)
    max_s1 = min(round(args.mu_max * args.n), 98*args.n)
    sk_bias.run_mcs(pdf, args.n, min_s1=min_s1, max_s1=max_s1)
elif args.command == 'run_unquantized_mcs':
    sk_bias.run_unquantized_mcs(args.n)
elif args.command == 'run_transit_mcs':
    sk_bias.run_transit_mcs(args.nt, args.ndish, args.brightness)
elif args.command == 'emit_code':
    binterp = sk_bias.BiasInterpolator()
    sk_bias.emit_code(binterp)
else:
    parser.print_help()
    sys.exit(2)

