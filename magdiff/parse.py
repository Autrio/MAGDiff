import argparse as ap


parser = ap.ArgumentParser()
parser.add_argument("-n", "--nworld", type=int, default=4, help="number of parallel worlds to simulate and render")
parser.add_argument("-v", "--nworld_rend", type=int, default=0, help="number of parallel worlds to render (must be <= nworld)")
parser.add_argument("--outfile", type=str, default="output.mp4", help="output video file name")

args = parser.parse_args()

NWORLD = args.nworld
NWORLD_REND = args.nworld_rend
OUTFILE = args.outfile