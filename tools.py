import re,os

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

                    line = re.sub('<[a-z\-=/\"\s]+>|\u3000','',line.strip())

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

    def get_labels(self):
        '''
        获取标签字典
        '''
        return {'usAAPL':1,'ftAP0':0}


if __name__ == '__main__':
    tool = Tools()

    tool.generate_valid_set('datas','train_datas')
