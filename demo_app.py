from flask import Flask, render_template, session, url_for, redirect, request
from flask.ext.wtf import Form
from flask.ext.bootstrap import Bootstrap
from wtforms import StringField, SubmitField, TextAreaField, SelectField
from wtforms.validators import Required

from page_utils import Pagination
from werkzeug.datastructures import ImmutableMultiDict

import json
import jieba
import os
import sys
from gensim.summarization.bm25 import BM25
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import datapath

# add by sz
import urllib.parse
import urllib.request

# add environment path
sys.path.append('/home/ubuntu/xunfei-demo/reformat') 
sys.path.append('/home/ubuntu/xunfei-demo/OpenNMT') 
sys.path.append('/home/ubuntu/xunfei-demo/correction')
sys.path.append('/home/ubuntu/xunfei-demo/summary') 

from reformat.task3 import format_rewrite
from OpenNMT.translate import translate_main
import correction.test_one 
from correction.test_one import correction_main
from summary.main import test
from baidu_search_crawler import search_main

# summary
example_inputs = []
with open('summary/data/demo.json', encoding='utf-8') as f:
    for line in f:
        dic = json.loads(line)
        example_inputs.append(dic['doc'])

# reformat
example_file_lst = ['原文1.txt', '原文2.txt', '原文3.txt', '原文4.txt']
corpus_input = []
for file_name in example_file_lst:
    with open('reformat/%s'%file_name, 'r', encoding='utf-8') as fr:
        corpus_input.append(''.join(fr.readlines()))

# BM25
corpus = []
index_dict = {}
f = open('doc_search/corpus.json', 'r').readlines()
text = [json.loads(t)['corpus'] for t in f]
for i in range(50000):
    line = json.loads(f[i])['corpus']
    for word in line:
        if word in index_dict.keys():
            index_dict[word].add(i)
        else:
            index_dict[word] = {i}
    corpus.append(line)

dictionary = Dictionary(text)
Model = BM25(corpus)
lda = LdaModel.load('doc_search/lda_model')

# template
file_name=os.listdir('static/doc_template/')
template_example_inputs = []
template_example_inputs_example = []
for file_id in file_name:
    with open('static/doc_template/' + file_id, 'r', encoding='utf-8') as fr:
        lines = fr.read()
        template_example_inputs.append(lines.strip())
for file_id in file_name:
    with open('static/doc_template_/'+file_id, 'r', encoding='utf-8') as fr:
        lines = fr.read()
        template_example_inputs_example.append(lines.strip())

def takeSecond(elem):
    return elem[1]

def search(query, model=Model, Index=index_dict, dic=dictionary, lda_model=lda):
    # query must be segmented with the form ['word1', 'word2', 'word3']
    rank = []
    query_corpus = dic.doc2bow(query)
    vector = lda_model[query_corpus]
    if len(vector) > 0 and len(query) <= 3:
        query += [dic[w[0]] for w in lda_model.get_topic_terms(vector[0][0], int(len(query) / 2) + 1)]
    doc_set = set()
    for word in query:
        doc_set = doc_set.union(Index[word])
    for id in doc_set:
        rank.append((id, model.get_score(query, int(id))))
    rank.sort(key=takeSecond, reverse=True)
    results = []
    for j in range(3):
        results.append(json.loads(f[rank[j][0]]))
    return results

class ReFormatForm(Form):
    post = TextAreaField('输入原文本')
    choice = SelectField('选择样例', choices=[
        ('0', '样例1'),
        ('1', '样例2'),
        ('2', '样例3'),
        ('3', '样例4')
    ], default='0')
    submit1 = SubmitField('填充')
    submit2 = SubmitField('提交')

class SummaryForm(Form):
    post = TextAreaField('会议对话')
    choice = SelectField('选择样例', choices=[
        ('0', '样例1'),
        ('1', '样例2'),
        ('2', '样例3'),
        ('3', '样例4')
    ], default='0')
    submit1 = SubmitField('填充')
    submit2 = SubmitField('提交')

class DocForm(Form):
    title_post = StringField('输入标题')
    tag_post = StringField('输入标签')
    choice1 = SelectField('选择标题', choices=[('0', '市财政积极推进预算绩效指标体系建设'), ('1', '天津政务网——实施最严格知识产权保护制度'), ('2', '环境空气质量标准'), ('3', '建设大数据创新平台')], default='0')
    choice2 = SelectField('选择标签', choices=[('0', '新闻'), ('1', '今日关注'), ('2', '环境 标准')], default='0')
    submit1 = SubmitField('填充')
    submit = SubmitField('提交')

class CorrectionForm(Form):
    post = TextAreaField('输入原文')
    choice = SelectField('选择样例', choices=[('0', '样例1'), ('1', '样例2'),('2', '样例3')], default='0')
    submit1 = SubmitField('填充')
    submit = SubmitField('智能检错')
    submit3 = SubmitField('语法检错')
    submit2 = SubmitField('别字校正')

class SearchForm(Form):
    post = StringField('输入关键词')
    choice = SelectField('选择关键词', choices=[('0', '人工智能调研报告'), ('1', '一带一路 专题报告'),('2','科大讯飞 2018年报')], default='0')
    submit1 = SubmitField('填充')
    submit = SubmitField('检索')

class TemplateForm(Form):
    choice = SelectField('选择模板', choices=[(str(i),file_name[i][0:-4]) for i in range(len(file_name))], default='0')
    submit1 = SubmitField('展示')

demo_app = Flask(__name__)
demo_app.config['SECRET_KEY'] = '961018961018'
bootstrap = Bootstrap(demo_app)

@demo_app.route('/', methods=['GET','POST'])
def home():
    return render_template('home.html')

@demo_app.route('/reformat', methods=['GET','POST'])
def reformat():
    post = None
    choice = None
    resp = None
    reformatForm = ReFormatForm()

    if reformatForm.submit1.data and reformatForm.validate_on_submit():
        choice = reformatForm.choice.data
        post = corpus_input[int(choice)]
        reformatForm.post.data = post

    if reformatForm.submit2.data and reformatForm.validate_on_submit():
        post = reformatForm.post.data
        if post == '':
            choice = reformatForm.choice.data
            post = corpus_input[int(choice)]
        with open('reformat/input.txt', 'w', encoding='utf-8') as fw:
            fw.write(post)
        post = post.split('\n')
        resp = format_rewrite()
        reformatForm.post.data = ''

    return render_template('reformat.html', form=reformatForm, post=post, resp=resp)

@demo_app.route('/summary', methods=['GET','POST'])
def summary():
    choice = None
    origin_text = None
    tgt_summary = None
    origin_text_list = None
    summaryForm = SummaryForm()

    if summaryForm.submit1.data and summaryForm.validate_on_submit():
        choice = summaryForm.choice.data
        summaryForm.post.data = example_inputs[int(choice)]
    
    if summaryForm.submit2.data and summaryForm.validate_on_submit():
        origin_text = summaryForm.post.data
        if origin_text == '':
            choice = summaryForm.choice.data
            origin_text = example_inputs[int(choice)]
        summary_indexs = test(origin_text)
        print(summary_indexs)
        idx = 0
        group = []
        tgt_summary = ''
        origin_text_list = []
        for sent in origin_text.split('\n'):
            sent = ''.join(sent.split())
            if ('：' in sent or ':' in sent) and group != []:
                origin_text_list.append(group)
                group = []
            if idx in summary_indexs:
                tgt_summary += sent
                group.append([1, sent])
            else:
                group.append([0, sent])
            idx += 1
        if group != []:
            origin_text_list.append(group)
        summaryForm.post.data = ''

    return render_template('summary.html', form=summaryForm, summary=tgt_summary, origin_text=origin_text_list)

@demo_app.route('/doc', methods=['GET','POST'])
def doc():
    title = None
    tags = None
    refs = None
    docs = None
    docForm = DocForm()
    choice1 = ''
    choice2 = ''
    text1=[ '市财政积极推进预算绩效指标体系建设', '天津政务网——实施最严格知识产权保护制度', '环境空气质量标准', '建设大数据创新平台']
    text2=['新闻', '今日关注', '环境 标准']
    if docForm.submit1.data and docForm.validate_on_submit():
        choice1 = text1[int(docForm.choice1.data)]
        choice2 = text2[int(docForm.choice2.data)]
        docForm.title_post.data = choice1
        docForm.tag_post.data = choice2
    if docForm.submit.data and docForm.validate_on_submit():
        title = jieba.lcut(docForm.title_post.data)
        tags = docForm.tag_post.data.split()
        srcs = [' '.join(tags + ['##'] + title)] * 2
        with open('OpenNMT/data/src-test.txt', 'w', encoding='utf-8') as fw:
            fw.write('\n'.join(srcs))

        search_results = search(title)
        refs = [r['corpus'] for r in search_results][:2]
        with open('OpenNMT/data/tgt-ref-test.txt', 'w', encoding='utf-8') as fw:
            fw.write('\n'.join([' '.join(r) for r in refs]))
        with open('OpenNMT/data/tgt-test.txt', 'w', encoding='utf-8') as fw:
            fw.write('\n'.join([' '.join(r) for r in refs]))
        
        translate_main()

        while not os.path.exists('OpenNMT/data/pred.txt'):
            continue
        with open('OpenNMT/data/pred.txt', 'r', encoding='utf-8') as fr:
            docs = [''.join(line.split()) for line in fr.readlines()]
        if os.path.exists('OpenNMT/data/pred.txt'):
            os.remove('OpenNMT/data/pred.txt')

    return render_template('doc.html', form=docForm, refs=[''.join(r) for r in refs] if refs is not None else refs, docs=docs)

# add by sz  
def errorfunc(files): # http://202.85.216.21:8095/get_error/ 输入分句好的list
    local_ip = '202.85.216.21'
    ports = [8095]
    port = ports[0]
    url = 'http://%s:%i/get_error/'%(local_ip, port)
    print(url) 
    print(files)
    #files = [str(f).replace('\ufeff', '') for f in files]
    formate = {
        "input_sentences":json.dumps(files)
    }
    # # python 2 版本
    # data = urllib.urlencode(formate) 
    # result = urllib2.urlopen(url=url, data=data.encode('utf-8'), timeout=10).read()
    # python 3 版本
    data = urllib.parse.urlencode(formate) 
    result = urllib.request.urlopen(url=url, data=data.encode('utf-8'), timeout=10).read()
    # print('result')
    print(json.loads(result)) 
    result = json.loads(result) 
    for i, this_sent in enumerate(result):
        if this_sent != []: 
            tmp = [all_wrong.insert(0, i+1) for all_wrong in this_sent] 
    print('result1')
    print(result)
    result = [r for r in result if r != [] ]
    print('result2')
    print(result)
    return result
'''
    result = json.loads(result)
    print(result) 
    if result[-1] == []:
        result = result[:-1] 
    print(result)  
    return result
'''

# add for <span class="error error-5">得</span>
def make_span(raw_post_list, return_json):
    # return_json [[[sentid, wordid, "", ""]], [[]], [[]]]
    for sent_ids in return_json:
        sent_ids = sent_ids[::-1]
        for word_ids in sent_ids:
            sentid, wordid, _, _ = word_ids
            tmp = raw_post_list[sentid-1]
            #raw_post_list[sentid-1] = tmp[:wordid] + '<span class="error error-5">' + tmp[wordid:wordid+1] + '</span>' + tmp[wordid+1:]
            raw_post_list[sentid-1] = tmp[:wordid] + '<span class="error error-'+str(wordid)+'">' + tmp[wordid:wordid+1] + '</span>' + tmp[wordid+1:]
    print('make_span:raw_post_list')
    print(raw_post_list)
    return raw_post_list

#add by gh&yhz
def errorfunc_grammar(files):
    ports = [8010]
    port = ports[0]
    sent_nums, char_nums = [], []
    local_ip = '202.85.216.20'
    json_result = None
    # for i in range(10):
    sents = {}
    # with codecs.open(path, 'r', encoding='utf-8') as f:
    # print('files')
    # print(files)
    for line in files:
        if len(line) > 0:
            lineparts = line
            # print('lineparts')
            # print(lineparts)
            # assert len(lineparts) == 2
            sents[lineparts[0]] = lineparts[1]
    char_nums.append(sum([len(sents[i]) for i in sents]))
    sent_nums.append(len(sents))
    sents_json = json.dumps(sents)
    # print(sents_json)
    try:
        # python 2 版本
        # result = urllib2.urlopen(url='http://%s:%i'%(local_ip, port), data=('texts=%s'%sents_json).encode('utf-8'), timeout=10)
        # python 3 版本
        result = urllib.request.urlopen(url='http://%s:%i' % (local_ip, port),
                                        data=('texts=%s' % sents_json).encode('utf-8'), timeout=10)
        json_result = eval(result.read())
    except:
        print('port %i is down' % port)
    return json_result


@demo_app.route('/correction', methods=['GET','POST'])
def correction():
    post = None
    resp = None
    mistake = None
    corForm = CorrectionForm()
    choice = ''
    text=[]
    for i in range(3):
        f = open('static/correction/correction_example'+str(i)+'.txt', 'r', encoding='utf-8')
        text.append(f.read())
        f.close()
    if corForm.submit1.data and corForm.validate_on_submit():
        choice = text[int(corForm.choice.data)]
        corForm.post.data=choice
    if corForm.submit.data and corForm.validate_on_submit():
        raw_post = corForm.post.data
        post = []

        # for sent in raw_post.split():
        #     lst = [w + '\tS-1' if i == 0 else w + '\tO' for i, w in enumerate(jieba.lcut(sent))]
        #     post.append('\n'.join(lst)) 
        
        # add by sz
        count = 0
        l = []
        for sent in raw_post.split():
            lst = [w + '\tS-1' if i == 0 else w + '\tO' for i, w in enumerate(jieba.lcut(sent))]
            l.extend(lst)
            count += len(lst)
            if count > 70: # 长一点再分句
                post.append('\n'.join(l)) 
                count = 0
                l = []
        post.append('\n'.join(l)) 

        with open('correction/officialdoc/raw.bmes', 'w', encoding='utf-8') as fw:
            # fw.write('\n\n' + '\n\n'.join(post) + '\n\n')
            fw.write('\n\n' + '|'.join(post) + '\n\n\n')
        resp = correction_main()
        mistake = ','.join([w[0] for w in resp if w[1]])
        resp = ''.join([w[0] for w in resp])

        corForm.post.data = ''
        print(mistake)
        print(resp)
    return_json=None
# add by sz
    raw_post_list=[]
    if corForm.submit2.data and corForm.validate_on_submit():
        raw_post = corForm.post.data
        # print(type(raw_post))
        # raw_post_list = raw_post.replace('。', '。|').split('|')
        raw_post_list = raw_post.split('\r\n')
        raw_post_list = [i for i in raw_post_list if i != '']
        #print(raw_post_list)
        new_list = []
        for sent in raw_post_list:
            if len(sent) > 127:
                sent = sent.replace('。', '。|').split('|')
                new_list.extend(sent)
            else:
                new_list.append(sent)
        assert len(new_list) >= len(raw_post_list)
        raw_post_list = new_list
        # print('raw_post_list11')
        # print(raw_post_list)
        raw_post_list = [f.replace('\ufeff', '') for f in raw_post_list]
        return_json = errorfunc(raw_post_list)
        # add for <span class="error error-5">得</span>
        raw_post_list = make_span(raw_post_list, return_json)
    
    # print(')))))))))))raw_post_list')
    # print(raw_post_list)    
        
    if (raw_post_list!=[] and raw_post_list[-1]==''):
        raw_post_list=raw_post_list[0:-1]
    # print('raw_post_list')
    # print(raw_post_list)

# add by yhz
    rp = []
    rp_ind = []
    er=0
    if corForm.submit3.data and corForm.validate_on_submit():
        if (corForm.post.data==''):
            er=1
        else:
            raw_post = corForm.post.data
            #print(raw_post)
            r_p_l = raw_post.split('\r\n')
            r_p_l = [i for i in r_p_l if i != '']
            new_list = []
            for sent in r_p_l:
                if len(sent) > 127:
                    sent = sent.replace('。', '。|').split('|')
                    new_list.extend(sent)
                else:
                    new_list.append(sent)
            assert len(new_list) >= len(r_p_l)
            r_p_l = new_list
            r_p_l = [f.replace('\ufeff', '') for f in r_p_l]
            r_p_l = [f for f in r_p_l if f!='']
            rp=r_p_l
            r_p_l = [[str(i),r_p_l[i]] for i in range(len(r_p_l))]
            # print('raw:\n')
            # print(raw_post_list)
            result_json = errorfunc_grammar(r_p_l)
            print(result_json)
            for cor in result_json:
                if (result_json[cor]=='wrong'):
                    rp[int(cor)]="<span style=\"color:red\">"+rp[int(cor)]+"</span>"
                    rp_ind.append(int(cor))
            print(rp)
    return render_template('correction.html',error=er, form=corForm, post=post, raw_text=raw_post_list, json=return_json, resp=resp, mistake=mistake,raw_of_grammar=rp, wrong_pos_of_grammar=rp_ind)

@demo_app.route('/search', methods=['GET', 'POST'])
def doc_search():
    post = None
    results = None
    html = None
    searchForm = SearchForm()
    text = ['人工智能调研报告',' 一带一路 专题报告','科大讯飞 2018年报']
    if searchForm.submit1.data and searchForm.validate_on_submit():
        choice = text[int(searchForm.choice.data)]
        searchForm.post.data = choice
    if searchForm.submit.data and searchForm.validate_on_submit():
        post = searchForm.post.data
        results = search_main(post)
        results.append(post)
        with open('doc_search.json', 'w') as fw:
            json.dump(results, fw)
    
    if results is None:
        if os.path.exists('doc_search.json'):
            with open('doc_search.json', 'r') as fr:
                results = json.load(fr)

    if results is not None:
        print(post)
        print(len(results))
        try:
            if post is None:
                current_page = request.args.get("page", 1)
                page_id = int(current_page) - 1
                args = request.args
                # 判断是刚打开页面还是在翻页
                if args == ImmutableMultiDict([]):
                    results = None
                post = results[-1]
            else:
                current_page = 1
                page_id = 0
                args = ImmutableMultiDict([('page', '1')])
            li = [i for i in range(1, len(results))]
            results = results[page_id * 10: (page_id + 1) * 10]
            print(args)
            pager_obj = Pagination(current_page, len(li), request.path, args, per_page_count=10)
            html = pager_obj.page_html()
        except:
            results = None
        
    return render_template("search.html", form=searchForm, post=post, html=html, results=results)
    # return render_template("search.html", form=searchForm, post=post, results=results)

@demo_app.route('/template', methods=['GET', 'POST'])
def template():
    templateForm = TemplateForm()
    sum=''
    hd=0
    ed1=0
    ed2=0
    ex=0
    _sum = ''
    _hd = 0
    _ed1 = 0
    _ed2 = 0
    _ex = 0
    if templateForm.submit1.data and templateForm.validate_on_submit():
        choice1=template_example_inputs[int(templateForm.choice.data)].split('\n')
        choice2=template_example_inputs_example[int(templateForm.choice.data)].split('\n')
        hd=choice1[0]
        if (choice1[-1][0]=='('):
            sum ='\n'.join(choice1[1:-3])
            ed1 =choice1[-3]
            ed2 =choice1[-2]
            ex = choice1[-1]
        else:
            sum = '\n'.join(choice1[1:-2])
            ed1 = choice1[-2]
            ed2 = choice1[-1]
            ex=''
        _hd = choice2[0]
        if (choice2[-1][0] == '('):
            _sum = '\n'.join(choice2[1:-3])
            _ed1 = choice2[-3]
            _ed2 = choice2[-2]
            _ex = choice2[-1]
        else:
            _sum = '\n'.join(choice2[1:-2])
            _ed1 = choice2[-2]
            _ed2 = choice2[-1]
            _ex = ''
    return render_template('template.html', form=templateForm,templ=sum,head=hd,end1=ed1,end2=ed2,extra=ex,
                           _templ=_sum,_head=_hd,_end1=_ed1,_end2=_ed2,_extra=_ex)

if __name__ == '__main__':
    demo_app.run(debug=True)