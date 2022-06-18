#!/bin/bash
#SBATCH -N 1
##SBATCH --reservation=actqrzwa6p_182 
#SBATCH --gres=dcu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=clean
#SBATCH -p kshdexclu05
#SBATCH --exclusive
#SBATCH --time=2000:00:00
##SBATCH --nodelist=j20r1n[14-19],j20r2n[00-01]
##SBATCH --nodelist=h04r1n06
##SBATCH -x e05r2n00
source env_hc_zjx.sh
#conda activate pytorch170_amd
which python3
hostfile=./$SLURM_JOB_ID
scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile}
rm `pwd`/hostfile-dl -f

for i in `cat $hostfile`
do
    echo ${i} slots=4 >> `pwd`/hostfile-dl-$SLURM_JOB_ID
done
np=$(cat $hostfile|sort|uniq |wc -l)

np=$(($np*1))

nodename=$(cat $hostfile |sed -n "1p")
echo $nodename
dist_url=`echo $nodename | awk '{print $1}'`
#for multi-gpu
echo mpirun -np $np --allow-run-as-root --hostfile hostfile-dl-$SLURM_JOB_ID  --bind-to none `pwd`/single_hc.sh $dist_url 
mpirun -np $np --allow-run-as-root --hostfile hostfile-dl-$SLURM_JOB_ID --bind-to none `pwd`/single_hc.sh $dist_url 
