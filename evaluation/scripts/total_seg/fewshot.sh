# 20-shot 
lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/baselines/random_init.yaml --trainer#callbacks=None --CONSTANTS#percentage=2 
pkill -9 -f lighter

lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/baselines/suprem_unet.yaml --trainer#callbacks=None --CONSTANTS#percentage=2
pkill -9 -f lighter

lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/baselines/suprem_segresnet.yaml --trainer#callbacks=None --CONSTANTS#percentage=2
pkill -9 -f lighter

lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/frameworks/conrecon.yaml --trainer#callbacks=None --CONSTANTS#percentage=2
pkill -9 -f lighter

lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/frameworks/simclr.yaml --trainer#callbacks=None --CONSTANTS#percentage=2
pkill -9 -f lighter

lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/frameworks/reconstruction.yaml --trainer#callbacks=None --CONSTANTS#percentage=2
pkill -9 -f lighter


# 10-shot
lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/baselines/random_init.yaml --trainer#callbacks=None --CONSTANTS#percentage=1 
pkill -9 -f lighter

lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/baselines/suprem_unet.yaml --trainer#callbacks=None --CONSTANTS#percentage=1
pkill -9 -f lighter

lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/baselines/suprem_segresnet.yaml --trainer#callbacks=None --CONSTANTS#percentage=1
pkill -9 -f lighter

lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/frameworks/conrecon.yaml --trainer#callbacks=None --CONSTANTS#percentage=1
pkill -9 -f lighter

lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/frameworks/simclr.yaml --trainer#callbacks=None --CONSTANTS#percentage=1
pkill -9 -f lighter

lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/frameworks/reconstruction.yaml --trainer#callbacks=None --CONSTANTS#percentage=1
pkill -9 -f lighter



# 5-shot
lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/baselines/random_init.yaml --trainer#callbacks=None --CONSTANTS#percentage=0.5
pkill -9 -f lighter

lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/baselines/suprem_unet.yaml --trainer#callbacks=None --CONSTANTS#percentage=0.5
pkill -9 -f lighter

lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/baselines/suprem_segresnet.yaml --trainer#callbacks=None --CONSTANTS#percentage=0.5
pkill -9 -f lighter

lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/frameworks/conrecon.yaml --trainer#callbacks=None --CONSTANTS#percentage=0.5
pkill -9 -f lighter

lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/frameworks/simclr.yaml --trainer#callbacks=None --CONSTANTS#percentage=0.5
pkill -9 -f lighter

lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/frameworks/reconstruction.yaml --trainer#callbacks=None --CONSTANTS#percentage=0.5
pkill -9 -f lighter