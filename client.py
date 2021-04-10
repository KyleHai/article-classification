import re,requests,json
import tensorflow as tf
import tokenization
import numpy as np

from common import Tools,convert_single_example,InputExample

tool = Tools()

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("max_seq_length", 80, "支持的文本序列最大长度")
flags.DEFINE_string("vocab_file", 'chinese_L-12_H-768_A-12/vocab.txt', "bert内置字典")

tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_file,
    do_lower_case=True)

def excute(line):
    '''
    执行函数
    :param line:
    :return:
    '''
    url_test = 'http://192.168.238.128:8508/v1/models/apple:predict'
    header = {"content-type": "application/json"}
    for line in re.split('[。！？]', line):
        line = line.strip('*')
        line = re.sub('\s', '', line)
        example = InputExample(guid=0,text=line,label=0)
        feature = convert_single_example(example,[0,1],FLAGS.max_seq_length,tokenizer)

        data = [{"input_ids":feature.input_ids,"input_mask":feature.input_mask,"segment_ids":feature.segment_ids,"label_ids":feature.label_id}]

        reps = requests.post(url=url_test,data=json.dumps({"instances": data}),headers=header)

        result = json.loads(reps.text)
        r = np.argmax(result['predictions'][0])

        if r == 1:
            print('“苹果”是指公司！')
        else:
            print('“苹果”是指水果！')


if __name__ == '__main__':
    sentence = "苹果手机真好用啊"
    excute(sentence)