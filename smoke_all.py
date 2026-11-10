"""Reduced real end-to-end pipeline; explicit fresh output directory required."""
import sys
from run_paper import main
if __name__=='__main__':
    sys.argv[1:1]=['--profile','smoke','--backend','cpu']
    main()
