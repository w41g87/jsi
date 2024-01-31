import argparse
import numpy as np
import scipy
import imageio
import jsi
from pathlib import Path

parser = argparse.ArgumentParser(description='Approximates an image\'s transparency channel through joint spectral intensity diagram.')
parser.add_argument('-i', '--input', help='Input previously trained file', type=Path)
parser.add_argument('-s', '--source', help='Input picture file path', type=Path)
parser.add_argument('-n', '--nodes', help='Result square JSI dimension', type=int)
parser.add_argument('-p', '--padding', default=0, help='JSI padding dimension to increase DOF, added to both sides of the JSI', type=int)
parser.add_argument('-o', '--output', help='Path for the output .npz file with trained parameters, the parameters are printed if this is not specified', type=Path)
parser.add_argument('-e', '--epochs', help='Number of iteratins to run', required=True, type=float)

args = parser.parse_args()
pred = None
loss = None
img = None

if args.input:
    with np.load(args.input.resolve()) as data:
        input = dict(data)
    # print(input)
    pred, loss = jsi.jsi_backprop(input, args.epochs, lr=1e-3)
elif args.source:
    if args.nodes:
        image = imageio.imread(args.source.resolve())
        target = np.zeros((args.nodes, args.nodes, 4), dtype=np.uint8)
        scipy.ndimage.zoom(image, args.nodes / max(image.shape), order=3, output=target)
        
        input = {
            'nodes' : args.nodes,
            'padding' : args.padding,
            'n_rings' : 5, 
            'target' : target[:, :, 3]
        }
        pred, loss, img = jsi.jsi_backprop(input, args.epochs, lr=1e-3)
    else:
        raise AttributeError('Missing Attribute: Output dimension unspecified, use -n to provide an output dimension')
else:
    parser.print_help()
    exit(0)

print('Final loss: ' + str(loss[-1]))
if args.output:
    np.savez(args.output.resolve(), **pred)
    jsi.pltSect(img, 0, 0, args.nodes, args.nodes)
    plt.savefig((args.output + '.png').resolve())
else:
    np.set_printoptions(formatter={'complexfloat': lambda x: "{0.real:0.3f} + {0.imag:0.3f}i".format(x)})
    print(pred)