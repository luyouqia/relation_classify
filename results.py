# coding: utf-8
import os
import re
map_dic = {'Cause-Effect(e1,e2)': '1', 'Cause-Effect(e2,e1)': '2',
           'Instrument-Agency(e1,e2)': '3', 'Instrument-Agency(e2,e1)': '4',
           'Product-Producer(e1,e2)': '5', 'Product-Producer(e2,e1)': '6',
           'Content-Container(e1,e2)': '7', 'Content-Container(e2,e1)': '8',
           'Entity-Origin(e1,e2)': '9', 'Entity-Origin(e2,e1)': '10',
           'Entity-Destination(e1,e2)': '11', 'Entity-Destination(e2,e1)': '12',
           'Component-Whole(e1,e2)': '13', 'Component-Whole(e2,e1)': '14',
           'Member-Collection(e1,e2)': '15', 'Member-Collection(e2,e1)': '16',
           'Message-Topic(e1,e2)': '17', 'Message-Topic(e2,e1)': '18',
           'Other': '0'}

map_new = {}
for key in map_dic:
    map_new[map_dic[key]] = key

if os.path.exists('results/'):
    os.system('rm -r results/')
os.mkdir('results')

os.system('export PATH=$PATH:/home/kbp2016/luyao/RC_test/SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2')

for root, dirs, files in os.walk('predictions'):
    for filename in files:
        pre_f = os.path.join(root, filename)
        if 'prediction' in filename:
            accuracy = pre_f.split('_')[1]
            f0 = open(
                '../SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2/proposed_answer_%s.txt' % accuracy, 'w')
            with open(pre_f, 'r') as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    line = lines[i].split('\t')[0]
                    f0.write(str(i + 1) + '\t' + map_new[line] + '\n')

            f1 = open(
                '../SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2/answer_key_%s.txt' % accuracy, 'w')
            with open(pre_f, 'r') as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    line = lines[i].split('\t')[1].strip()
                    f1.write(str(i + 1) + '\t' + map_new[line] + '\n')
            f0.close()
            f1.close()

            score_dic = '../SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2/'
            os.system('perl {dic}semeval2010_task8_scorer-v1.2.pl {dic}proposed_answer_{s}.txt {dic}answer_key_{s}.txt > {dic}result_scores_{s}.txt'.format(
                s=accuracy, dic=score_dic))
            os.system(
                'mv {dic}result_scores_{accuracy}.txt results/'.format(dic=score_dic, accuracy=accuracy))
            os.system(
                'rm {dic}proposed_answer_{s}.txt'.format(s=accuracy, dic=score_dic))
            os.system(
                'rm {dic}answer_key_{s}.txt'.format(s=accuracy, dic=score_dic))

for root, dirs, files in os.walk('results/'):
    for filename in files:
        res_f = os.path.join(root, filename)
        text = open(res_f).read()
        score = re.findall(
            r'<<< The official score is \(9\+1\)-way evaluation with directionality taken into account\: macro-averaged F1 \= (.*?) >>>', text)[0]
        print filename, ':', score
