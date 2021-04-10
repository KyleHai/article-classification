from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections, os, re
import modeling
import optimization
import tokenization
import tensorflow as tf
import numpy as np

from common import Tools, InputExample, InputFeatures, convert_single_example, apple_data_Processor, create_model

tool = Tools()

flags = tf.flags
FLAGS = flags.FLAGS

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

def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    '''
    将样本转换为tf_record文件
    :param examples:
    :param label_list:
    :param max_seq_length:
    :param tokenizer:
    :param output_file:
    :return:
    '''
    # tf_recordwriter的输出文件路径
    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        # 将单个样本转换为特征输入
        feature = convert_single_example(example, label_list,
                                         max_seq_length, tokenizer)

        # 创建tf.train.Feature()整数特征
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        # 先定义一个有序字典，用来存储特征
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])
        # 将特征字典先转换为训练特征实例，再转换为训练样本实例
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        # 将训练样本写入到tf_record中，先将tf_example序列化为字符串，然后写入到tf_record中
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    '''
    从input_file的tf_record中，读取信息
    :param input_file:
    :param seq_length:
    :param is_training:
    :param drop_remainder:
    :return:
    '''
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """解码一个tfrecord样本"""
        example = tf.parse_single_example(record, name_to_features)

        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """输入函数"""
        batch_size = params["batch_size"]

        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))
        return d

    return input_fn

def _truncate_seq_pair(tokens, max_length):
    """裁剪超过最大长度的tokens"""
    while True:
        if len(tokens) <= max_length:
            break
        tokens.pop()

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings=False):
    '''
    创建一个分类模型
    :param bert_config:
    :param is_training:
    :param input_ids:
    :param input_mask:
    :param segment_ids:
    :param labels:
    :param num_labels:
    :param use_one_hot_embeddings:
    :return:
    '''
    # 创建bert模型
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)
    # 得到bert模型的输出
    output_layer = model.get_pooled_output()
    # 获取隐藏层的大小
    hidden_size = output_layer.shape[-1].value
    # 定义分类器权重系数矩阵
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    # 定义分类器偏置向量
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # 设置0.1的dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        # 求logits
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        # 进行归一化求概率
        probabilities = tf.nn.softmax(logits, axis=-1)
        # 进行归一化求似然概率
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        # 对label进行one—hot向量化
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        # 求每个样本的loss
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        # 求当前batch的平均loss
        loss = tf.reduce_mean(per_example_loss)
        # 返回相应的值
        return (loss, per_example_loss, logits, probabilities)


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
        # 打印相关日志信息
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        # 从tf_record中读取相应特征数据
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        # 判定是否为训练模式
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        # 创建bert+softmax的分类模型，并返回loss,per_example_loss,logits,probabilities
        (total_loss, per_example_loss, logits, probabilities) = create_model(
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
                predictions={"probabilities": probabilities},
            )
        else:
            output_spec = None
            raise Exception("传给model_fn中的mode参数出错")

        return output_spec

    return model_fn


def input_fn_builder(features, seq_length, is_training, drop_remainder):
    '''
    输入函数生成器
    :param features:
    :param seq_length:
    :param is_training:
    :param drop_remainder:
    :return:
    '''
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """将输入样本列表转换成一个特征列表。"""
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


def main(_):
    # 设置日志级别
    tf.logging.set_verbosity(tf.logging.INFO)
    # 从bert配置文件中获取配置参数
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    # 判定FLAGS设定的最大长度是否超过bert预训练模型所能支持的最大长度
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "无法使用长度 %d，因为bert模型所能支持的最大长度是%d。" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))
    # 创建输出文件夹
    tf.gfile.MakeDirs(FLAGS.output_dir)
    # 声明一个数据处理器实例
    processor = apple_data_Processor()
    # 获取训练集的标签
    label_list = processor.get_labels()
    # 实例化一个基于词典的token切分器
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file)

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    params = {}

    if FLAGS.do_train:
        params['batch_size'] = FLAGS.train_batch_size
        # 获取训练集数据
        all_examples = processor.get_examples(FLAGS.data_dir)
        train_examples, test_examples = processor.generate_train_test(all_examples)

        # 根据训练数据条数、批大小、训练的周期，计算出要执行的步数
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        # 计算出warmup的周期步数
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # 生成一个前向模型
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
    )

    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.output_dir, params=params)

    if FLAGS.do_train:
        # 训练样本数据转换为tf_record后的保存地址
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        # 将样本特征数据转换为tf_record文件
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        # 向日志中添加训练数据信息
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        # 从tf_record文件中读取要训练的数据
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        # 训练样本和训练步数传入评估器的train函数中，开始进行训练
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)


if __name__ == "__main__":
    # 训练
    flags.DEFINE_bool("do_train", True, "执行训练模式")
    flags.DEFINE_bool("do_predict", False, "执行预测模式")
    flags.DEFINE_string("data_dir", 'train_datas', "数据目录")

    tf.app.run()
