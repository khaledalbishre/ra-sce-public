#!/bin/bash
set -e
FAISS="artifacts/indexes/msmarco_contriever.index"
IDS="artifacts/indexes/msmarco_ids.npy"
MODEL="artifacts/models/contriever"
RESULTS="artifacts/results"

echo "=== 1. Main experiments ==="
python benchmark_ra_sce_v4.py \
  --index_path $FAISS --id_path $IDS \
  --model_name $MODEL --results_dir $RESULTS \
  --dataset_key TREC_DL_19 2>&1 | tee $RESULTS/main_dl19.log

python benchmark_ra_sce_v4.py \
  --index_path $FAISS --id_path $IDS \
  --model_name $MODEL --results_dir $RESULTS \
  --dataset_key TREC_DL_20 2>&1 | tee $RESULTS/main_dl20.log

echo "=== 2. Rho sweep DL19 ==="
for rho in 0.1 0.2 0.3 0.4; do
python benchmark_ra_sce_v4.py \
  --index_path $FAISS --id_path $IDS \
  --model_name $MODEL --results_dir $RESULTS/rho_${rho} \
  --dataset_key TREC_DL_19 --rho $rho \
  2>&1 | tee $RESULTS/rho_${rho}_dl19.log
done

echo "=== 3. Rho sweep DL20 ==="
for rho in 0.1 0.2 0.3 0.4; do
python benchmark_ra_sce_v4.py \
  --index_path $FAISS --id_path $IDS \
  --model_name $MODEL --results_dir $RESULTS/rho_${rho} \
  --dataset_key TREC_DL_20 --rho $rho \
  2>&1 | tee $RESULTS/rho_${rho}_dl20.log
done

echo "=== 4. Ablations DL19 ==="
python benchmark_ra_sce_v4.py \
  --index_path $FAISS --id_path $IDS \
  --model_name $MODEL --results_dir $RESULTS/abl_no_agree \
  --dataset_key TREC_DL_19 --agreement_gate False \
  2>&1 | tee $RESULTS/abl_no_agree_dl19.log

python benchmark_ra_sce_v4.py \
  --index_path $FAISS --id_path $IDS \
  --model_name $MODEL --results_dir $RESULTS/abl_no_conf \
  --dataset_key TREC_DL_19 --confidence_lock False \
  2>&1 | tee $RESULTS/abl_no_conf_dl19.log

python benchmark_ra_sce_v4.py \
  --index_path $FAISS --id_path $IDS \
  --model_name $MODEL --results_dir $RESULTS/abl_no_topn \
  --dataset_key TREC_DL_19 --topn_lock False \
  2>&1 | tee $RESULTS/abl_no_topn_dl19.log

python benchmark_ra_sce_v4.py \
  --index_path $FAISS --id_path $IDS \
  --model_name $MODEL --results_dir $RESULTS/abl_all_off \
  --dataset_key TREC_DL_19 \
  --agreement_gate False --confidence_lock False --topn_lock False \
  2>&1 | tee $RESULTS/abl_all_off_dl19.log

echo "=== 5. Ablations DL20 ==="
python benchmark_ra_sce_v4.py \
  --index_path $FAISS --id_path $IDS \
  --model_name $MODEL --results_dir $RESULTS/abl_no_agree \
  --dataset_key TREC_DL_20 --agreement_gate False \
  2>&1 | tee $RESULTS/abl_no_agree_dl20.log

python benchmark_ra_sce_v4.py \
  --index_path $FAISS --id_path $IDS \
  --model_name $MODEL --results_dir $RESULTS/abl_no_conf \
  --dataset_key TREC_DL_20 --confidence_lock False \
  2>&1 | tee $RESULTS/abl_no_conf_dl20.log

python benchmark_ra_sce_v4.py \
  --index_path $FAISS --id_path $IDS \
  --model_name $MODEL --results_dir $RESULTS/abl_no_topn \
  --dataset_key TREC_DL_20 --topn_lock False \
  2>&1 | tee $RESULTS/abl_no_topn_dl20.log

python benchmark_ra_sce_v4.py \
  --index_path $FAISS --id_path $IDS \
  --model_name $MODEL --results_dir $RESULTS/abl_all_off \
  --dataset_key TREC_DL_20 \
  --agreement_gate False --confidence_lock False --topn_lock False \
  2>&1 | tee $RESULTS/abl_all_off_dl20.log

echo "=== ALL DONE ==="
echo "Results summary:"
cat $RESULTS/ra_sce_v4_summary.csv
