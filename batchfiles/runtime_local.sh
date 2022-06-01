for i in {1..2}
do
    python ../runtime_experiment.py -s lorenz -t $i -n runtime_local_test -p 3 -w
#    python ../runtime_experiment.py -s mhd -t $i -n runtime_local_mio -p 3 -w
#    python ../runtime_experiment.py -s hopf -t $i -n runtime_local_mio -p 3 -w

#    python ../runtime_experiment.py -s lorenz -t $i -n runtime_local_mio -p 5 -w
#    python ../runtime_experiment.py -s hopf -t $i -n runtime_local_mio -p 5 -w
#    python ../runtime_experiment.py -s mhd -t $i -n runtime_local_mio -p 5 -w
done