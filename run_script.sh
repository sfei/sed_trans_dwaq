

# RH: Previous runs in this folder used this stompy. Appears to have no local
#     changes, dates back to github master circa April 2024.
#     No obvious incompatibilities introduced in commits between then and
#     now (2025-01-30).
export PYTHONPATH="/opt/software/rusty/stompy/newest_commit/stompy"

#DELFT_BASE="/opt/anaconda3/envs/dfm_t141798optO3"
DELFT_BASE="/opt/software/delft/delft3dfm-lnx64-2026.01-057d2f7/lnx64"

# set a bunch of environment variables
export DELFT_BIN=$DELFT_BASE/bin
export DELFT_SRC=$DELFT_BASE/bin   # note: this is a place holder, not used unless running from source
export DELWAQ2_PATH=$DELFT_BASE/bin/delwaq2
export DELFT_SHARE=$DELFT_BASE/share/delft3d
export LD_LIBRARY_PATH=$DELFT_BASE/lib:$LD_LIBRARY_PATH

# create inputs 
/opt/anaconda3/envs/delft_env/bin/python -u RunLauncher.py &> delwaq1.out

# execute run
#/opt/anaconda3/envs/dfm_t141798optO3/bin/delwaq2 sfbay_dynamo000.inp -openpb lib_sedmod.so > delwaq2.out
#/opt/anaconda3/envs/dfm_t141798optO3/bin/delwaq2 sfbay_dynamo000.inp > delwaq2.out
/opt/software/delft/delft3dfm-lnx64-2026.01-057d2f7/lnx64/bin/delwaq2 sfbay_dynamo000.inp > delwaq2.out

# remove input files that are not needed for QC and debugging
# rm *.wrk *.seg *.flo *.vol *.par *.are *.srf *.poi *.csv *.dsp *.len *.ts *.mon *.bnd *.atr
# rm -r forcing
