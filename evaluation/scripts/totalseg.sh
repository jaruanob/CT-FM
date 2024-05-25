export WANDB_ENTITY=aim-harvard


######################### Decoder-only ############################
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_unet.yaml
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_segresnet.yaml
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/frameworks/conrecon.yaml



######################### Few-shot train samples ############################
# 20-shot 
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/random_init.yaml --trainer#callbacks=None --CONSTANTS#percentage=2 
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_unet.yaml --trainer#callbacks=None --CONSTANTS#percentage=2
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_segresnet.yaml --trainer#callbacks=None --CONSTANTS#percentage=2
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/frameworks/conrecon.yaml --trainer#callbacks=None --CONSTANTS#percentage=2


# 10-shot
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/random_init.yaml --trainer#callbacks=None --CONSTANTS#percentage=1 
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_unet.yaml --trainer#callbacks=None --CONSTANTS#percentage=1
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_segresnet.yaml --trainer#callbacks=None --CONSTANTS#percentage=1
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/frameworks/conrecon.yaml --trainer#callbacks=None --CONSTANTS#percentage=1

# 5-shot
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/random_init.yaml --trainer#callbacks=None --CONSTANTS#percentage=0.5
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_unet.yaml --trainer#callbacks=None --CONSTANTS#percentage=0.5
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_segresnet.yaml --trainer#callbacks=None --CONSTANTS#percentage=0.5
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/frameworks/conrecon.yaml --trainer#callbacks=None --CONSTANTS#percentage=0.5



######################### Few-shot annotation groups ############################
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/random_init.yaml --trainer#callbacks=None --CONSTANTS#group="cardiac"
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_unet.yaml --trainer#callbacks=None --CONSTANTS#group="cardiac"
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_segresnet.yaml --trainer#callbacks=None --CONSTANTS#group="cardiac"
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/frameworks/conrecon.yaml --trainer#callbacks=None --CONSTANTS#group="cardiac"


lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_unet.yaml --trainer#callbacks=None --CONSTANTS#group="organs"
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/random_init.yaml --trainer#callbacks=None --CONSTANTS#group="organs"
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_segresnet.yaml --trainer#callbacks=None --CONSTANTS#group="organs"
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/frameworks/conrecon.yaml --trainer#callbacks=None --CONSTANTS#group="organs"

lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/random_init.yaml --trainer#callbacks=None --CONSTANTS#group="muscles"
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_unet.yaml --trainer#callbacks=None --CONSTANTS#group="muscles"
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_segresnet.yaml --trainer#callbacks=None --CONSTANTS#group="muscles"
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/frameworks/conrecon.yaml --trainer#callbacks=None --CONSTANTS#group="muscles"

lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/random_init.yaml --trainer#callbacks=None --CONSTANTS#group="vertebra"
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_unet.yaml --trainer#callbacks=None --CONSTANTS#group="vertebra"
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_segresnet.yaml --trainer#callbacks=None --CONSTANTS#group="vertebra"
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/frameworks/conrecon.yaml --trainer#callbacks=None --CONSTANTS#group="vertebra"

lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/random_init.yaml --trainer#callbacks=None --CONSTANTS#group="muscles"
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_unet.yaml --trainer#callbacks=None --CONSTANTS#group="muscles"
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_segresnet.yaml --trainer#callbacks=None --CONSTANTS#group="muscles"
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/frameworks/conrecon.yaml --trainer#callbacks=None --CONSTANTS#group="muscles"


######################### Full ############################
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/random_init.yaml --trainer#callbacks=None
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_unet.yaml --trainer#callbacks=None
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_segresnet.yaml --trainer#callbacks=None
lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/frameworks/conrecon.yaml --trainer#callbacks=None
