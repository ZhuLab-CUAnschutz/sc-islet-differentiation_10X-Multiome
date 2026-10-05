import os
import h5py
import argparse
import numpy as np

# Sibling of /cellar/users/aklie/projects/ML4GLand/chrombpnet/scripts/modisco_to_pfm.py
# Identical trim/threshold/seqlet logic, but reads neg_patterns instead of pos_patterns.

def trim_ppm(ppm, t=0.45, min_length=3, flank=0):
    # trim matrix to first and last bp that have
    # p>=threshold
    maxes = np.max(ppm,-1)
    maxes = np.where(maxes>=t)

    # if no bases with prob>t or too small:
    if (len(maxes[0])==0) or (maxes[0][-1]+1-maxes[0][0]<min_length):
        return None

    return ppm[max(maxes[0][0]-flank, 0):maxes[0][-1]+1+flank]

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--modisco_h5py", required=True, type=str)
    parser.add_argument("-o", "--output_dir", required=True, type=str)
    parser.add_argument("-t", "--threshold", default=0.45, type=float)
    parser.add_argument("-ml", "--min_length", default=3, type=int, help="Min length of acceptable motif")
    parser.add_argument("-f", "--flank_add", default=0, type=int, help="Add this on either side of motif after trimming")
    parser.add_argument("-s", "--min_seqlets", default=100, type=int, help="Minimum seqlets associated with motif")
    parser.add_argument("-n", "--normalize", default=False, action='store_true', help="PPM (probability) instead of PFM (frequency)" )
    parser.add_argument("-op", "--output_prefix", default="", type=str, help="Prefix for output files")
    args = parser.parse_args()

    f = h5py.File(args.modisco_h5py, 'r')

    if 'neg_patterns' not in f:
        print("WROTE 0 neg PFMs (no neg_patterns group)")
        f.close()
        raise SystemExit(0)

    num_patterns = len(f['neg_patterns'])-1

    written = 0
    for i in range(num_patterns):
        trimmed_ppm = trim_ppm(f['neg_patterns']['pattern_{}'.format(i)]['sequence'],
                t=args.threshold, min_length=args.min_length, flank=args.flank_add)

        if trimmed_ppm is not None:
            num_seqlets = len(f['neg_patterns']['pattern_{}'.format(i)]['seqlets']['sequence'])


            if num_seqlets >= args.min_seqlets:
                pfm = (trimmed_ppm*num_seqlets).astype(int)

                if args.normalize:
                    pfm = (pfm+1)/np.sum(pfm, axis=1, keepdims=True)
                    np.savetxt(os.path.join(args.output_dir, "{}.pattern_{}.pfm".format(args.output_prefix, i)),
                        pfm, fmt='%.2f', delimiter='\t')

                else:
                   np.savetxt(os.path.join(args.output_dir, "{}.pattern_{}.pfm".format(args.output_prefix, i)),
                        pfm, fmt='%d', delimiter='\t')
                written += 1

    print("WROTE {} neg PFMs".format(written))
    f.close()
