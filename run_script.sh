# activate conda environment
#conda activate dwaq_env


# RH: Previous runs in this folder used this stompy. Appears to have no local
#     changes, dates back to github master circa April 2024.
#     No obvious incompatibilities introduced in commits between then and
#     now (2025-01-30).
export PYTHONPATH="/opt/software/rusty/stompy/newest_commit/stompy"
export LD_LIBRARY_PATH=/opt/anaconda3/envs/dfm_t141798optO3/lib:$LD_LIBRARY_PATH

# create inputs 
/opt/anaconda3/envs/delft_env/bin/python -u RunLauncher.py &> delwaq1.out

# execute run
#/opt/anaconda3/envs/dfm_t141798optO3/bin/delwaq2 sfbay_dynamo000.inp -openpb lib_sedmod.so > delwaq2.out
/opt/anaconda3/envs/dfm_t141798optO3/bin/delwaq2 sfbay_dynamo000.inp > delwaq2.out

# remove input files that are not needed for QC and debugging
# rm *.wrk *.seg *.flo *.vol *.par *.are *.srf *.poi *.csv *.dsp *.len *.ts *.mon *.bnd *.atr
# rm -r forcing
