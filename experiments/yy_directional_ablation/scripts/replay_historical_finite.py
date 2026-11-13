"""Re-render Figure 8 from all 60 original E5 packets, without refitting."""
import argparse
import json
from pathlib import Path

from plot_historical_finite import plot, read_packets, summarize, write_csv
from verify_historical_finite import ROOT, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Fresh output directory')
    output = parser.parse_args().output.resolve()
    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    report = audit()
    repeats = read_packets(ROOT/'data/finite/packets')
    summary = summarize(repeats)
    output.mkdir(parents=True)
    write_csv(output/'repeat_values_dual_metrics.csv', repeats)
    write_csv(output/'summary_dual_metrics.csv', summary)
    plot(summary, output/'figure_e5_finite_sample_dual_metrics')
    (output/'AUDIT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(packets=60,summary_rows=24,new_fits=False,output=str(output))))


if __name__=='__main__':main()
