import numpy as np, torch, pyfaidx, seqpro as sp
from bpnetlite.bpnet import BPNet
from tangermeme.predict import predict
G="/carter/users/aklie/data/ref/genomes/hg38/hg38.fa"
M="/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/models/DE/fold_0/chrombpnet/0.5/models/chrombpnet_nobias.h5"
genome=pyfaidx.Fasta(G); model=BPNet.from_chrombpnet(M).cuda().eval()
chrom,pos,a1,a2="chr17",40539784,"G","A"; half=1057
seq=genome[chrom][pos-half:pos+half].seq.upper()
ref=seq[:half]+a1+seq[half+1:]   # FORCE allele1
alt=seq[:half]+a2+seq[half+1:]   # FORCE allele2
def cnt(s):
    f=np.expand_dims(sp.ohe(s,alphabet=sp.DNA),0).transpose(0,2,1)
    r=np.expand_dims(sp.ohe(sp.reverse_complement(s,sp.DNA),alphabet=sp.DNA),0).transpose(0,2,1)
    _,cf=predict(model,torch.tensor(f).float().cuda()); _,cr=predict(model,torch.tensor(r).float().cuda())
    return float(np.exp(np.squeeze(cf.cpu().detach().numpy()))),float(np.exp(np.squeeze(cr.cpu().detach().numpy())))
rf,rr=cnt(ref); af,ar=cnt(alt)
print(f"FORCED alleles: REF(G) mean={(rf+rr)/2:.1f}  ALT(A) mean={(af+ar)/2:.1f}")
print(f"logfc fwd-only={np.log(af/rf):+.3f}  fwd+rev={np.log(((af+ar)/2)/((rf+rr)/2)):+.3f}")
print("file logfc = -0.924 (ref>alt, loss)")
