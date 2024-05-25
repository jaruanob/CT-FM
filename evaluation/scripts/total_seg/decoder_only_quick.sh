
# Lightweight decoder with upsampling layers non-trainable (halves the number of params)
# lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/frameworks/conrecon.yaml --system#model#trunk#model#upsample_mode="nontrainable" --CONSTANTS#percentage=5
# lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/frameworks/simclr.yaml --system#model#trunk#model#upsample_mode="nontrainable" --CONSTANTS#percentage=5
# lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/frameworks/reconstruction.yaml --system#model#trunk#model#upsample_mode="nontrainable" --CONSTANTS#percentage=5


# lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/frameworks/simclr.yaml --CONSTANTS#percentage=5
# lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/frameworks/reconstruction.yaml --CONSTANTS#percentage=5 


# lighter fit --config_file=./evaluation/totalseg.yaml,./evaluation/frameworks/conrecon_v2.yaml --CONSTANTS#percentage=5 --CONSTANTS#project="ct_fm_quick_seg_eval"
# lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/baselines/random_init.yaml --CONSTANTS#percentage=5 --CONSTANTS#project="ct_fm_quick_seg_eval"
