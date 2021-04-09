import os,esm

fw = open('data_error.txt','w',encoding='utf-8')
for root,_,files in os.walk('datas'):
    usapple = ['苹果公司','苹果商店','股市','市盈率','上涨','乔布斯','库克','ipad','iphone','mac']
    index_usapple = esm.Index()
    for i in usapple:
        index_usapple.enter(i)
    index_usapple.fix()

    ftap0 = ['水果','期货','养殖','畜牧','养鸡']
    index_ftap0 = esm.Index()
    for i in ftap0:
        index_ftap0.enter(i)
    index_ftap0.fix()

    for file in files:
        path = os.path.join(root,file)
        fr = open(path,'r',encoding='utf-8')
        for line in fr:
            if 'ftAP0' in file:
                if index_usapple.query(line):
                    fw.write(file+'\n')
                    break
            elif 'usAAPL' in file:
                if index_ftap0.query(line):
                    fw.write(file+'\n')
                    break
fw.close()