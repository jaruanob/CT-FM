#!/bin/bash

# Weights paths
ct_fm_path="/mnt/data16/ibro/IDC_SSL_CT/runs/fm/checkpoints/CT_FM_SimCLR_SegResNetDS/epoch=449-step=225000-v1.ckpt"
suprem_path="/home/ibrahim/Projects/CT_FM/baselines/SuPreM_UNet/supervised_suprem_unet_2100.pth"

datetime="2024-10-06_11-53-46"

export WANDB_ENTITY=aim-harvard

# Test - SinoCT

# 100% data
lighter predict --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=100 --trainer#callbacks#0#until_epoch=0 --vars#name="ct_fm_100pct" --vars#datetime=${datetime} --vars#mode=predict_sinoct --system#model#trunk#ckpt_path=$ct_fm_path --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s
lighter predict --config=./evaluation/sinoct.yaml,./evaluation/baselines/unetencoder_suprem.yaml --vars#percentage=100 --trainer#callbacks#0#until_epoch=0 --vars#name="suprem_unetencoder_100pct" --vars#datetime=${datetime} --vars#mode=predict_sinoct --system#model#trunk#ckpt_path=$suprem_path --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s
lighter predict --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=100 --trainer#callbacks#0#until_epoch=0 --vars#name="baseline_100pct" --vars#datetime=${datetime} --vars#mode=predict_sinoct --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s

# 10% data
lighter predict --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=10 --trainer#callbacks#0#until_epoch=0 --vars#name="ct_fm_10pct" --vars#datetime=${datetime} --vars#mode=predict_sinoct --system#model#trunk#ckpt_path=$ct_fm_path --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s
lighter predict --config=./evaluation/sinoct.yaml,./evaluation/baselines/unetencoder_suprem.yaml --vars#percentage=10 --trainer#callbacks#0#until_epoch=0 --vars#name="suprem_unetencoder_10pct" --vars#datetime=${datetime} --vars#mode=predict_sinoct --system#model#trunk#ckpt_path=$suprem_path --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s
lighter predict --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=10 --trainer#callbacks#0#until_epoch=0 --vars#name="baseline_10pct" --vars#datetime=${datetime} --vars#mode=predict_sinoct --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s

# 5% data
lighter predict --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=5 --trainer#callbacks#0#until_epoch=0 --vars#name="ct_fm_5pct" --vars#datetime=${datetime} --vars#mode=predict_sinoct --system#model#trunk#ckpt_path=$ct_fm_path --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s
lighter predict --config=./evaluation/sinoct.yaml,./evaluation/baselines/unetencoder_suprem.yaml --vars#percentage=5 --trainer#callbacks#0#until_epoch=0 --vars#name="suprem_unetencoder_5pct" --vars#datetime=${datetime} --vars#mode=predict_sinoct --system#model#trunk#ckpt_path=$suprem_path --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s
lighter predict --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=5 --trainer#callbacks#0#until_epoch=0 --vars#name="baseline_5pct" --vars#datetime=${datetime} --vars#mode=predict_sinoct --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s

# 2% data
lighter predict --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=2 --trainer#callbacks#0#until_epoch=0 --vars#name="ct_fm_2pct" --vars#datetime=${datetime} --vars#mode=predict_sinoct --system#model#trunk#ckpt_path=$ct_fm_path --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s
lighter predict --config=./evaluation/sinoct.yaml,./evaluation/baselines/unetencoder_suprem.yaml --vars#percentage=2 --trainer#callbacks#0#until_epoch=0 --vars#name="suprem_unetencoder_2pct" --vars#datetime=${datetime} --vars#mode=predict_sinoct --system#model#trunk#ckpt_path=$suprem_path --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s
lighter predict --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=2 --trainer#callbacks#0#until_epoch=0 --vars#name="baseline_2pct" --vars#datetime=${datetime} --vars#mode=predict_sinoct --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s

# Test - CQ500

# 100% data
lighter predict --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=100 --vars#name="ct_fm_100pct" --vars#datetime=${datetime} --vars#mode=predict_cq500 --system#model#trunk#ckpt_path=$ct_fm_path
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s
lighter predict --config=./evaluation/cq500.yaml,./evaluation/baselines/unetencoder_suprem.yaml --vars#percentage=100 --vars#name="suprem_unetencoder_100pct" --vars#datetime=${datetime} --vars#mode=predict_cq500 --system#model#trunk#ckpt_path=$suprem_path
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s
lighter predict --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=100 --vars#name="baseline_100pct" --vars#datetime=${datetime} --vars#mode=predict_cq500
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s

# 10% data
lighter predict --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=10 --vars#name="ct_fm_10pct" --vars#datetime=${datetime} --vars#mode=predict_cq500 --system#model#trunk#ckpt_path=$ct_fm_path
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s
lighter predict --config=./evaluation/cq500.yaml,./evaluation/baselines/unetencoder_suprem.yaml --vars#percentage=10 --vars#name="suprem_unetencoder_10pct" --vars#datetime=${datetime} --vars#mode=predict_cq500 --system#model#trunk#ckpt_path=$suprem_path
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s
lighter predict --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=10 --vars#name="baseline_10pct" --vars#datetime=${datetime} --vars#mode=predict_cq500
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s

# 5% data
lighter predict --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=5 --vars#name="ct_fm_5pct" --vars#datetime=${datetime} --vars#mode=predict_cq500 --system#model#trunk#ckpt_path=$ct_fm_path
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s
lighter predict --config=./evaluation/cq500.yaml,./evaluation/baselines/unetencoder_suprem.yaml --vars#percentage=5 --vars#name="suprem_unetencoder_5pct" --vars#datetime=${datetime} --vars#mode=predict_cq500 --system#model#trunk#ckpt_path=$suprem_path
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s
lighter predict --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=5 --vars#name="baseline_5pct" --vars#datetime=${datetime} --vars#mode=predict_cq500
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s

# 2% data
lighter predict --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=2 --vars#name="ct_fm_2pct" --vars#datetime=${datetime} --vars#mode=predict_cq500 --system#model#trunk#ckpt_path=$ct_fm_path
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s
lighter predict --config=./evaluation/cq500.yaml,./evaluation/baselines/unetencoder_suprem.yaml --vars#percentage=2 --vars#name="suprem_unetencoder_2pct" --vars#datetime=${datetime} --vars#mode=predict_cq500 --system#model#trunk#ckpt_path=$suprem_path
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s
lighter predict --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=2 --vars#name="baseline_2pct" --vars#datetime=${datetime} --vars#mode=predict_cq500
pkill -9 -f lighter
pkill -9 -f wandb
sleep 5s