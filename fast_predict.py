import re, os
from tokenization import FullTokenizer, validate_case_matches_checkpoint

try:
    from . import modeling
except:
    import modeling

import optimization
import tokenization
import collections
import numpy as np
import pandas as pd

from common import Tools, InputExample, InputFeatures, convert_single_example, apple_data_Processor, create_model

import tensorflow as tf
from tensorflow.python.estimator.estimator import Estimator
from sklearn.model_selection import train_test_split

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
                predictions={"probabilities": probabilities, "logits": logits, "labels": labels,
                             "one_hot_labels": one_hot_labels, 'outlayer': output_layer},
            )
        else:
            output_spec = None
            raise Exception("传给model_fn中的mode参数出错")

        return output_spec

    return model_fn


class FastPredict(object):
    def __init__(self, label):
        self.label = label
        self.closed = False
        self.first_run = True
        self.tokenizer = FullTokenizer(
            vocab_file=FLAGS.vocab_file,
            do_lower_case=True)
        self.init_checkpoint = FLAGS.init_checkpoint
        self.seq_length = FLAGS.max_seq_length
        self.text = None
        self.num_examples = None
        self.predictions = None
        self.estimator = self.get_estimator()

    def get_estimator(self):
        # 检查节点是否一致
        validate_case_matches_checkpoint(True, self.init_checkpoint)
        # 载入bert自定义配置
        bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
        if FLAGS.max_seq_length > bert_config.max_position_embeddings:
            if FLAGS.max_seq_length > bert_config.max_position_embeddings:
                raise ValueError(
                    "无法使用长度 %d，因为bert模型所能支持的最大长度是%d。" %
                    (self.seq_length, bert_config.max_position_embeddings))

        model_fn = model_fn_builder(  # 估计器函数，提供Estimator使用的model_fn，内部使用EstimatorSpec构建的
            bert_config=bert_config,
            num_labels=len(self.label),
            init_checkpoint=self.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=None,
            num_warmup_steps=None,
        )

        estimator = Estimator(  # 实例化估计器
            model_fn=model_fn,
            warm_start_from=self.init_checkpoint  # 新增预热
        )
        return estimator

    def get_feature(self, text):
        example = InputExample(0, text, self.label[0])
        feature = convert_single_example(example, self.label, self.seq_length, self.tokenizer)
        return feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_id

    def create_generator(self):
        """构建生成器"""
        while not self.closed:
            self.num_examples = len(self.text)
            features = (self.get_feature(f) for f in self.text)
            yield dict(zip(("input_ids", "input_mask", "segment_ids", "label_ids", "is_real_example"), zip(*features)))

    def input_fn_builder(self):
        """用于预测单独对预测数据进行创建，不基于文件数据"""
        dataset = tf.data.Dataset.from_generator(
            self.create_generator,
            output_types={'input_ids': tf.int64,
                          'input_mask': tf.int64,
                          'segment_ids': tf.int64,
                          'label_ids': tf.int64},
            output_shapes={
                'label_ids': (None),
                'input_ids': (None, None),
                'input_mask': (None, None),
                'segment_ids': (None, None)}
        )
        return dataset

    def predict(self, text):
        self.text = [text]
        if self.first_run:
            self.predictions = self.estimator.predict(
                input_fn=self.input_fn_builder, yield_single_examples=False)
            self.first_run = False
        prediction = next(self.predictions)

        return [self.label[i] for i in prediction["probabilities"].argmax(axis=1)]

    def close(self):
        self.closed = True


def predict_score_fn(fast, path):
    '''
    预测投票
    :param fast:
    :param path:
    :return:
    '''
    score_dict = dict()
    line_info = []

    for line in tool.format_article(path):
        anw = fast.predict(line)
        anwser = anw[0]
        if anwser == 0:
            if 0 not in score_dict:
                score_dict[0] = 1
            else:
                score_dict[0] = score_dict[0] + 1
        else:
            if 1 not in score_dict:
                score_dict[1] = 1
            else:
                score_dict[1] = score_dict[1] + 1

        line_info.append((line, len(line), anwser))

    if not score_dict or not line_info:
        # 全文预测
        with open(path,'r',encoding='utf-8') as fr:
            for line in fr.readlines():
                for line in re.split('<p>|</p>',line.lower()):
                    if not line.strip():
                        continue

                    line = re.sub('<[a-z\-=/\":\'‘’：_;；\s]+>|\u3000','',line.strip())

                    for line in re.split('[。！？]',line):
                        line = line.strip('*')
                        line = re.sub('\s','',line)
                        tmp_ans = fast.predict(line)[0]

                        if tmp_ans == 0:
                            if 0 not in score_dict:
                                score_dict[0] = 1
                            else:
                                score_dict[0] = score_dict[0] + 1
                        else:
                            if 1 not in score_dict:
                                score_dict[1] = 1
                            else:
                                score_dict[1] = score_dict[1] + 1

                        line_info.append((line, len(line), tmp_ans))

    if 0 in score_dict and 1 not in score_dict:
        art_label = 0
    elif 1 in score_dict and 0 not in score_dict:
        art_label = 1
    elif max(score_dict[0], score_dict[1]) == min(score_dict[0], score_dict[1]):
        line_info = sorted(line_info, key=lambda x: x[1], reverse=True)
        art_label = line_info[0][-1]
    else:
        score_dict = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        art_label = score_dict[0][0]

    return art_label


if __name__ == '__main__':
    flags.DEFINE_bool("do_train", False, "执行训练模式")
    flags.DEFINE_bool("do_predict", True, "执行预测模式")
    # 设置要用的模型版本
    flags.DEFINE_string("pretrain_checkpoint", "out_put/model.ckpt-7210", "预训练好的模型")

    # 设置测试目录
    flags.DEFINE_string("data_dir", 'test_datas', "数据目录")
    # 设置测试结果保存路径
    save_predict_path = 'result/all_test_md7210.xlsx'

    FLAGS.init_checkpoint = FLAGS.pretrain_checkpoint
    # 实例一个快速预测对象
    fast = FastPredict(label=[0, 1])
    # 获取目录下的数据
    paths = tool.get_predict_datas(FLAGS.data_dir)

    # 从训练数据中随机抽取0.1比例进行预测，仅限于进行训练集的验证时,进行测试集验证时，需要注释掉下一行
    #_,paths = train_test_split(paths,train_size=0.9,test_size=0.1)
    #print('test examples size ',len(paths))

    # 对每个文件进行预测
    file_names = []
    true_lbs = []
    pred_lbs = []
    for ct,pt in enumerate(paths):
        filename = os.path.basename(pt)
        # 获取真实标签，ftAP0用0表示，usAAPL用1表示。
        if 'ftAP0' in filename:
            true_label = 0
        elif 'usAAPL' in filename:
            true_label = 1
        else:
            true_label = 'None'

        art_label = predict_score_fn(fast, pt)

        file_names.append(filename)
        true_lbs.append(true_label)
        pred_lbs.append(art_label)

        print('正在预测第{}篇.'.format(ct))
        # if ct == 6:
        #     break

    tmp = {'filename':file_names,'true_label':true_lbs,'pred_label':pred_lbs}
    df = pd.DataFrame(data=tmp,columns=['filename','true_label','pred_label'])

    df.to_excel(save_predict_path)
