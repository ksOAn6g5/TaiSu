#!/bin/bash
#SBATCH -N 25
#SBATCH --gres=dcu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --job-name=RN_w+o
#SBATCH -o RN_web+ofa.%j
#SBATCH --exclusive
#SBATCH -p test


#### SBATCH --nodelist=h01r2n[00-19]
source env_hc.sh
hostfile=./$SLURM_JOB_ID
scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile}
rm `pwd`/hostfile-dl -f
for i in `cat $hostfile`
do
    echo ${i} slots=4 >> `pwd`/hostfile-dl-$SLURM_JOB_ID
done
np=$(cat $hostfile|sort|uniq |wc -l)
np=$(($np*4))
nodename=$(cat $hostfile |sed -n "1p")
echo $nodename
dist_url=`echo $nodename | awk '{print $1}'`
#for multi-gpu
echo mpirun -np $np --allow-run-as-root --hostfile hostfile-dl-$SLURM_JOB_ID  --bind-to none `pwd`/single_hc.sh $dist_url 
mpirun -np $np --allow-run-as-root --hostfile hostfile-dl-$SLURM_JOB_ID --bind-to none `pwd`/single_hc.sh $dist_url 
