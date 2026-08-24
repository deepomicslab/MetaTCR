"""Reconstruct full-length TCRβ `full_seq` from CDR3 + V gene + J gene.

Adapted from TITAN's cdr3_to_full_seq.py
(https://github.com/PaccMann/TITAN/blob/main/scripts/cdr3_to_full_seq.py): the CDR3 is
aligned (Bio.pairwise2) to the IMGT V- and J-segment sequences and extended in both
directions, so `full_seq = V region + CDR3 + J region`. A row is `Failure` if the V
segment is not found or the result is <= 50 aa. Input files run in parallel; every input
column is kept and only `full_seq` is added (TSV output; `--drop-failures` drops failures).

Usage:
    python cdr3_to_full_seq.py IN1_filt.tsv IN2_filt.tsv ... \
        --segment-dir TCR_gene_segment_data \
        --v-header vMaxResolved --j-header jMaxResolved --cdr3-header aminoAcid \
        --out-dir OUTDIR --in-suffix _filt.tsv --out-suffix _filt_full.tsv \
        --procs 16 [--drop-failures]
"""
import os
import csv
import argparse
from multiprocessing import Pool

from Bio import SeqIO
from Bio import pairwise2

# ---- module-level FASTA cache, populated once per worker by init_worker ----
V_RECORDS = None   # list[(record.id, str(record.seq))] in FASTA file order
J_RECORDS = None


def rename_Vseg(Vname):
    if len(Vname) > 1 and Vname[1] == 'C':
        Vname = 'TRB' + Vname[4:]
        if Vname[Vname.find('-') + 1] == '0':
            Vname = Vname[:(Vname.find('-') +
                            1)] + Vname[(Vname.find('-') + 2):]
        if Vname[Vname.find('V') + 1] == '0':
            Vname = Vname[:(Vname.find('V') +
                            1)] + Vname[(Vname.find('V') + 2):]
    return Vname


def rename_Jseg(Jname):
    if len(Jname) > 1 and Jname[1] == 'C':
        Jname = 'TRB' + Jname[4:]

        if Jname[Jname.find('-') + 1] == '0':
            Jname = Jname[:(Jname.find('-') +
                            1)] + Jname[(Jname.find('-') + 2):]
        if Jname[Jname.find('J') + 1] == '0':
            Jname = Jname[:(Jname.find('J') +
                            1)] + Jname[(Jname.find('J') + 2):]
    return Jname


def to_full_seq(Vname, Jname, CDR3):
    """Reconstruct one full_seq; the V/J records come from the cached V_RECORDS /
    J_RECORDS globals (parsed once per worker) instead of a per-row SeqIO.parse."""
    if not str(CDR3).strip():
        # empty CDR3 -> nothing to align (pairwise2 returns an empty list and would
        # crash on [0]); treat as unreconstructable -> Failure.
        return '', False, False
    foundV = False
    foundJ = False
    Vseq = ''
    Jseq = ''
    for i in range(1, 9):
        if f'-0{i}' in Vname:
            Vname = Vname.replace(f'-0{i}', f'-{i}')
            break
    if ':' in Jname:
        Jname = Jname.replace(':', '-')
    for Vrecord_id, Vrecord_seq in V_RECORDS:
        if type(Vname) != str or Vname == 'unresolved':
            print('Vname not string but ', Vname, type(Vname))
            Vseq = ''
        else:
            ## Deal with inconsistent naming conventions of segments
            Vname_adapted = rename_Vseg(Vname)
            if Vname_adapted in Vrecord_id:
                Vseq = Vrecord_seq
                foundV = True
            elif '-' in Vname_adapted:
                Vname_adapted = Vname_adapted.split('-')[0]
                if Vname_adapted in Vrecord_id:
                    Vseq = Vrecord_seq
                    foundV = True

    for Jrecord_id, Jrecord_seq in J_RECORDS:
        if type(Jname) != str or Jname == 'unresolved':
            print('Jname not string but ', Jname, type(Jname))
            Jseq = ''
        else:
            ## Deal with inconsistent naming conventions of segments
            Jname_adapted = rename_Jseg(Jname)
            if Jname_adapted in Jrecord_id:
                Jseq = Jrecord_seq
                foundJ = True
    if Vseq is None:
        foundV = False
    if foundV and Vseq != '':
        ## Align end of V segment to CDR3
        alignment = pairwise2.align.globalxx(
            Vseq[-5:],  # last five amino acids overlap with CDR3
            CDR3,
            one_alignment_only=True,
            penalize_end_gaps=(False, False)
        )[0]
        best = list(alignment[1])

        ## Deal with deletions
        if best[0] == '-' and best[1] == '-':
            best[0] = Vseq[-5]
            best[1] = Vseq[-4]
        if best[0] == '-':
            best[0] = Vseq[-5]

        # remove all left over -
        best = "".join(list(filter(lambda a: a != '-', best)))
    else:
        best = CDR3

    ## Align CDR3 sequence to start of J segment
    if Jseq != '':
        alignment = pairwise2.align.globalxx(
            best,
            Jseq,
            one_alignment_only=True,
            penalize_end_gaps=(False, False)
        )[0]

        # From last position, replace - with J segment amino acid
        # until first amino acid of CDR3 sequence is reached
        best = list(alignment[0])[::-1]
        firstletter = 0
        for i, aa in enumerate(best):
            if aa == '-' and firstletter == 0:
                best[i] = list(alignment[1])[::-1][i]
            else:
                firstletter = 1

        # remove all left over -
        best = "".join(list(filter(lambda a: a != '-', best[::-1])))

    full_sequence = Vseq[:-5] + best

    return full_sequence, foundV, foundJ


def init_worker(segment_dir):
    """Parse the V/J FASTAs once, storing (id, str(seq)) in file order."""
    global V_RECORDS, J_RECORDS
    V_RECORDS = [(r.id, str(r.seq)) for r in
                 SeqIO.parse(os.path.join(segment_dir, 'V_segment_sequences.fasta'), "fasta")]
    J_RECORDS = [(r.id, str(r.seq)) for r in
                 SeqIO.parse(os.path.join(segment_dir, 'J_segment_sequences.fasta'), "fasta")]


def process_one(task):
    """Reconstruct one file, preserving every input column BYTE-FOR-BYTE and touching
    only the full_seq column. Reads raw fields (no float/number round-trip), recomputes
    full_seq from the V/J/CDR3 fields, and writes the original
    fields back verbatim. Output is TSV; an existing full_seq column is overwritten in
    place, otherwise one is appended."""
    input_path, output_path, v_header, j_header, cdr3_header, drop_failures = task
    in_delim = ',' if os.path.splitext(input_path)[1] == '.csv' else '\t'
    with open(input_path, newline='', encoding='utf-8-sig') as fh:
        reader = csv.reader(fh, delimiter=in_delim)
        header = next(reader)
        rows = [row for row in reader if row]

    vi, ji, ci = header.index(v_header), header.index(j_header), header.index(cdr3_header)
    has_full = 'full_seq' in header
    fi = header.index('full_seq') if has_full else None
    out_header = header if has_full else header + ['full_seq']

    num_input = len(rows)
    num_failures = 0
    out_rows = []
    for row in rows:
        full_seq, foundV, foundJ = to_full_seq(row[vi], row[ji], row[ci])
        value = str(full_seq) if (foundV and len(full_seq) > 50) else 'Failure'
        if value == 'Failure':
            num_failures += 1
        if has_full:
            row[fi] = value
        else:
            row = row + [value]
        if drop_failures and value == 'Failure':
            continue
        out_rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh, delimiter='\t', lineterminator='\n')
        writer.writerow(out_header)
        writer.writerows(out_rows)
    return input_path, output_path, num_input, num_failures, len(out_rows)


def output_path_for(input_path, args):
    base = os.path.basename(input_path)
    if base.endswith(args.in_suffix):
        out_base = base[:-len(args.in_suffix)] + args.out_suffix
    else:
        stem = base[:-4] if base.lower().endswith(('.tsv', '.csv')) else base
        out_base = stem + '_full.tsv'
    out_dir = args.out_dir if args.out_dir else os.path.dirname(input_path)
    return os.path.join(out_dir, out_base)


def parse_args():
    p = argparse.ArgumentParser(
        description="Accelerated cdr3_to_full_seq (FASTA-cache + file-level MP); "
                    "reconstruction byte-identical to cdr3_to_full_seq.py.")
    p.add_argument("inputs", nargs="+", help="input .tsv/.csv repertoire files")
    p.add_argument("--segment-dir", required=True,
                   help="directory with V_segment_sequences.fasta and J_segment_sequences.fasta")
    p.add_argument("--v-header", default="vMaxResolved")
    p.add_argument("--j-header", default="jMaxResolved")
    p.add_argument("--cdr3-header", default="aminoAcid")
    p.add_argument("--out-dir", default=None, help="output dir (default: alongside each input)")
    p.add_argument("--in-suffix", default="_filt.tsv")
    p.add_argument("--out-suffix", default="_filt_full.tsv")
    p.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--drop-failures", action="store_true",
                   help="drop rows whose reconstruction failed (project _filt_full.tsv convention); "
                        "default keeps them marked 'Failure')")
    return p.parse_args()


def main():
    args = parse_args()
    tasks = []
    for input_path in args.inputs:
        output_path = output_path_for(input_path, args)
        if os.path.exists(output_path) and not args.overwrite:
            print(f"skip (exists): {output_path}", flush=True)
            continue
        tasks.append((input_path, output_path, args.v_header, args.j_header,
                      args.cdr3_header, args.drop_failures))

    if not tasks:
        print("nothing to do")
        return

    procs = min(args.procs, len(tasks))
    with Pool(processes=procs, initializer=init_worker, initargs=(args.segment_dir,)) as pool:
        results = pool.map(process_one, tasks)

    total_in = sum(r[2] for r in results)
    failures = sum(r[3] for r in results)
    total_out = sum(r[4] for r in results)
    pct = (100 * failures / total_in) if total_in else 0.0
    print(f"DONE files={len(results)} input_rows={total_in} failures={failures} ({pct:.2f}%) "
          f"output_rows={total_out} drop_failures={args.drop_failures} procs={procs}", flush=True)


if __name__ == '__main__':
    main()
