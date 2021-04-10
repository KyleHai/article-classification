import re,os
import tokenization
import modeling
import tensorflow as tf
from sklearn.model_selection import train_test_split

class InputExample():
    '''
    样本类.
    '''

    def __init__(self, guid, text, label):
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """特征类"""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class apple_data_Processor(object):
    """apple数据集处理器"""

    def __init__(self):
        pass

    def get_examples(self, data_dir):
        '''
        从data_dir目录中，加载训练数据
        :param data_dir:
        :return:
        '''
        examples = []
        for root, _, files in os.walk(data_dir):
            guid = 0
            for file in files:
                for line in open(os.path.join(data_dir, file), 'r', encoding='utf-8'):
                    if not line.strip():
                        continue

                    line_list = re.split('\t', line.strip())

                    try:
                        text = tokenization.convert_to_unicode(line_list[0])
                        label = tokenization.convert_to_unicode(line_list[1])
                    except:
                        raise Exception('数据格式异常，文本与标签应"反斜杠t"进行切分。')

                    examples.append(InputExample(guid=guid, text=text, label=label))
                    guid += 1

        return examples

    def generate_train_test(self, examples):
        '''
        生成训练集和验证集
        '''
        train, test = train_test_split(examples, train_size=1, test_size=0, shuffle=True)
        return train, test

    def get_labels(self):
        '''
        获取标签数据
        :return:
        '''
        return [0, 1]

def convert_single_example(example, label_list, max_seq_length,
                           tokenizer):
    '''
    转换一个样本为一个特征样例
    :param example:
    :param label_list:
    :param max_seq_length:
    :param tokenizer:
    :return:
    '''
    # 形成标签映射，0对0、1对1。
    label_map = dict()
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens = tokenizer.tokenize(example.text)

    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0:(max_seq_length - 2)]

    new_tokens = []
    segment_ids = []
    new_tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens:
        new_tokens.append(token)
        segment_ids.append(0)
    new_tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(new_tokens)

    input_mask = [1] * len(input_ids)

    # 使用0进行padding.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    label_id = label_map[int(example.label)]

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=[label_id],
        is_real_example=True)

    return feature

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
    batch_size = output_layer.shape[0].value
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
        return (loss, per_example_loss, logits, probabilities, labels, one_hot_labels,output_layer)

def predict_input_fn_builder(one_feature):
    '''
    从input_file的tf_record中，读取信息
    :param sentence:
    :param seq_length:
    :param drop_remainder:
    :return:
    '''
    def input_fn():
        name_to_features = {
            "input_ids": tf.constant(value=[one_feature.input_ids], dtype=tf.int64),
            "input_mask": tf.constant(value=[one_feature.input_mask], dtype=tf.int64),
            "segment_ids": tf.constant(value=[one_feature.segment_ids], dtype=tf.int64),
            "label_ids": tf.constant(value=[one_feature.label_id], dtype=tf.int64),
            "is_real_example": tf.FixedLenFeature([], tf.int64),
        }
        return name_to_features

    return input_fn

class Tools():
    '''
    工具类
    '''
    def __init__(self):
        pass

    def format_article(self,filename,keys=['苹果','apple']):
        '''
        格式化篇章数据，提取有效数据到列表中
        '''
        rs = list()
        compiler = re.compile('|'.join(keys))
        with open(filename,'r',encoding='utf-8') as fr:
            for line in fr.readlines():
                for line in re.split('<p>|</p>',line.lower()):
                    if not compiler.search(line):
                        continue

                    line = re.sub('<[a-z\-=/\":\'‘’：_;；\s]+>|\u3000','',line.strip())

                    for line in re.split('[。！？]',line):
                        if not compiler.search(line):
                            continue
                        line = line.strip('*')
                        line = re.sub('\s','',line)
                        rs.append(line)
        return rs

    def generate_valid_set(self,data_dir,train_dir):
        '''
        遍历data_dir目录，生成有效数据集
        '''
        for _,_,files in os.walk(data_dir):
            for file in files:
                if re.search('usAAPL',file):
                    label = 1
                elif re.search('ftAP0',file):
                    label = 0
                else:
                    continue

                formated = self.format_article(os.path.join(data_dir,file))
                with open(os.path.join(train_dir,file),'w',encoding='utf-8') as fw:
                    for line in formated:
                        fw.write('{sentence}\t{label}\n'.format(sentence=line,label=label))

    def get_predict_datas(self,data_dir):
        '''
        从目录中获取预测数据
        :param data_dir:
        :return:
        '''
        paths = []
        for root,_,files in os.walk(data_dir):
            for file in files:
                paths.append(os.path.join(root,file))
        return paths

    def get_label_dict(self):
        '''
        获取标签字典
        '''
        return {0:'ftAP0',1:'usAAPL'}


if __name__ == '__main__':
    tool = Tools()

    tool.generate_valid_set('datas','train_datas')
