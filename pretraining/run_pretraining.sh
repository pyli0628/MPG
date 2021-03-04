#!/bin/bash

train_batch_size=${1:-8192}
learning_rate=${2:-"6e-3"}
precision=${3:-"fp32"}
num_gpus=${4:-8}
warmup_proportion=${5:-"0.2843"}
train_steps=${6:-8393}
save_checkpoint_steps=${7:-200}
resume_training=${8:-"false"}
create_logfile=${9:-"true"}
accumulate_gradients=${10:-"true"}
gradient_accumulation_steps=${11:-128}
seed=${12:-666}
job_name=${13:-"mol_pretraining"}
allreduce_post_accumulation=${14:-"true"}
allreduce_post_accumulation_fp16=${15:-"true"}
train_batch_size_phase2=${16:-4096}
learning_rate_phase2=${17:-"4e-3"}
warmup_proportion_phase2=${18:-"0.128"}
train_steps_phase2=${19:-1563}
gradient_accumulation_steps_phase2=${20:-512}

CODEDIR=${23:-"./pretraining"}
init_checkpoint=${24:-"None"}
RESULTS_DIR=$CODEDIR/results
CHECKPOINTS_DIR=$RESULTS_DIR/checkpoints

mkdir -p $CHECKPOINTS_DIR



PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--fp16"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
elif [ "$precision" = "tf32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

ACCUMULATE_GRADIENTS=""
if [ "$accumulate_gradients" == "true" ] ; then
   ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps"
fi

CHECKPOINT=""
if [ "$resume_training" == "true" ] ; then
   CHECKPOINT="--resume_from_checkpoint"
fi

ALL_REDUCE_POST_ACCUMULATION=""
if [ "$allreduce_post_accumulation" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION="--allreduce_post_accumulation"
fi

ALL_REDUCE_POST_ACCUMULATION_FP16=""
if [ "$allreduce_post_accumulation_fp16" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION_FP16="--allreduce_post_accumulation_fp16"
fi

INIT_CHECKPOINT=""
if [ "$init_checkpoint" != "None" ] ; then
   INIT_CHECKPOINT="--init_checkpoint=$init_checkpoint"
fi

echo $DATA_DIR_PHASE1
INPUT_DIR=$DATA_DIR_PHASE1
CMD=" $CODEDIR/run_pretraining.py"

CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --train_batch_size=$train_batch_size"
CMD+=" --max_seq_length=128"
CMD+=" --max_predictions_per_seq=20"
CMD+=" --max_steps=$train_steps"
CMD+=" --warmup_proportion=$warmup_proportion"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate"
CMD+=" --seed=$seed"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION_FP16"
CMD+=" $INIT_CHECKPOINT"
CMD+=" --do_train"
CMD+=" --json-summary ${RESULTS_DIR}/dllogger.json "

CMD="python3 -m torch.distributed.launch --nproc_per_node=$num_gpus $CMD"


if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size \* $num_gpus)
  printf -v TAG "pyt_bert_pretraining_phase1_%s_gbs%d" "$precision" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$RESULTS_DIR/$job_name.$TAG.$DATESTAMP.log
  printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi

set +x

echo "finished pretraining"

#Start Phase2


