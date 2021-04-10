import tensorflow as tf

import os
import modeling, tokenization, optimization
from common import InputExample, convert_single_example, create_model

flags = tf.flags
FLAGS = flags.FLAGS

# 需要设置的必要参数
# 需要设置的必要参数
flags.DEFINE_string("bert_config_file", 'chinese_L-12_H-768_A-12/bert_config.json', "bert参数配置文件")

flags.DEFINE_string("task_name", "apple_classify", "任务名称")

flags.DEFINE_string("vocab_file", 'chinese_L-12_H-768_A-12/vocab.txt', "bert内置字典")

flags.DEFINE_string("output_dir", 'out_put', "模型输出文件")

flags.DEFINE_string("init_checkpoint", "chinese_L-12_H-768_A-12/bert_model.ckpt", "初始化bert模型")

flags.DEFINE_integer("max_seq_length", 80, "支持的文本序列最大长度")

flags.DEFINE_integer("train_batch_size", 32, "训练时batch大小")

flags.DEFINE_integer("predict_batch_size", 1, "预测时batch大小")

flags.DEFINE_float("learning_rate", 5e-5, "Adam的学习率")

flags.DEFINE_float("num_train_epochs", 10, "总的训练周期数")

flags.DEFINE_float("warmup_proportion", 0.1, "执行warmup的训练步数")

flags.DEFINE_integer("save_checkpoints_steps", 1000, "保存模型的步间隔")

flags.DEFINE_integer("iterations_per_loop", 1000, "每经过1000次进行一次召回")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_file,
    do_lower_case=True)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps):
    '''
    模型函数生成器，返回一个Estimator。
    :param bert_config:
    :param num_labels:
    :param init_checkpoint:
    :param learning_rate:
    :param num_train_steps:
    :param num_warmup_steps:
    :return:
    '''

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        '''
        定义model_fn，需要传入四个参数：features,labels,mode,params,同时还要在这里面实现训练、评估、预测的程序。
        :param features: tf.features
        :param labels:
        :param mode:
        :param params:
        :return:
        '''
        # 从tf_record中读取相应特征数据
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        # 判定是否为训练模式
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        # 创建bert+softmax的分类模型，并返回loss,per_example_loss,logits,probabilities
        (total_loss, per_example_loss, logits, probabilities, labels, one_hot_labels, output_layer) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels)
        # 获取全部的可变量
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # 如果为训练模式，创建优化器
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=probabilities,
                export_outputs={
                    "predict":
                        tf.estimator.export.PredictOutput(probabilities)
                }
            )
        else:
            output_spec = None
            raise Exception("传给model_fn中的mode参数出错")

        return output_spec

    return model_fn

def serving_input_fn():
    input_ids = tf.placeholder(name='input_ids',dtype=tf.int64,shape=[None,None])
    input_mask = tf.placeholder(name='input_mask',dtype=tf.int64,shape=[None,None])
    segment_ids = tf.placeholder(name='segment_ids',dtype=tf.int64,shape=[None,None])
    label_ids = tf.placeholder(name='label_ids',dtype=tf.int64,shape=[None])

    receiver_tensors = {
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
        'label_ids': label_ids
    }

    return tf.estimator.export.ServingInputReceiver(features=receiver_tensors, receiver_tensors=receiver_tensors)



def main(estimator_base_path):
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    model_fn = model_fn_builder(  # 估计器函数，提供Estimator使用的model_fn，内部使用EstimatorSpec构建的
        bert_config=bert_config,
        num_labels=len([0,1]),
        init_checkpoint=FLAGS.pretrain_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=None,
        num_warmup_steps=None,
    )

    estimator = tf.estimator.Estimator(  # 实例化估计器
        model_fn=model_fn,
        warm_start_from=FLAGS.pretrain_checkpoint  # 新增预热
    )

    estimator.export_saved_model(export_dir_base=estimator_base_path, serving_input_receiver_fn=serving_input_fn,checkpoint_path=FLAGS.pretrain_checkpoint)

if __name__ == '__main__':
    flags.DEFINE_bool("do_train", False, "执行训练模式")
    flags.DEFINE_bool("do_predict", True, "执行预测模式")
    # 设置要用的模型版本
    flags.DEFINE_string("pretrain_checkpoint", "out_put/model.ckpt-7210", "预训练好的模型")

    tmpdir = 'savedmodel'
    estimator_base_path = os.path.join(tmpdir, 'from_estimator')
    main(estimator_base_path)
