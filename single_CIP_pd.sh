PARAMFILE=pd_params2/p$1
number=$(($2+1))
p1=`(sed -n "$number"p $PARAMFILE) | awk '{print $1}'`
p2=`(sed -n "$number"p $PARAMFILE) | awk '{print $2}'`
p3=`(sed -n "$number"p $PARAMFILE) | awk '{print $3}'`
p4=`(sed -n "$number"p $PARAMFILE) | awk '{print $4}'`

source /local/jrenton/anaconda2/my_anaconda.sh

nice -19 python run_CIP_parallel_pd.py $p1 $p2 $p3 $p4 11 
