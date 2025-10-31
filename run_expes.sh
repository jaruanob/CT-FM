lighter fit --config=./experiments/fm/base_mod.yaml,\
./experiments/fm/frameworks/intrasample_simclr.yaml,\
./experiments/fm/backbones/segresenc.yaml

lighter fit --config=./experiments/fm/base_server.yaml,\
./experiments/fm/frameworks/intrasample_simclr.yaml,\
./experiments/fm/backbones/segresenc.yaml

lighter fit --config=./experiments/fm/base_mod.yaml,\
./experiments/fm/frameworks/intrasample_simclr.yaml,\
./experiments/fm/backbones/segresenc_pretrained.yaml


lighter fit --config=./experiments/fm/base_server.yaml --trainer#devices=[2],\
./experiments/fm/frameworks/intrasample_simclr.yaml,\
./experiments/fm/backbones/segresenc.yaml