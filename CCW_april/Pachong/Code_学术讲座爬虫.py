import pandas as pd
import requests
import re


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 SE 2.X MetaSr 1.0"}

#文本清洗
def clean(txt):
    for i in ['</th>','<td>','<p>','</p>','<div>','</div>','\r\n','<','>']:
        txt = txt.replace(i, '')
    return txt

# 爬取主讲人介绍与讲座简介 url_i = 'https://wise.xmu.edu.cn/'+ results[0][0][6:]
def intro(url):
    response_i = requests.get(url, headers = headers)
    response_i.encoding='utf-8'
    text_i = response_i.text
    regix = '''<div.*?>.*?<div class="content">.*?<div class="article_content">.*?主讲人简介：(.*?)</td>.*?讲座简介：(.*?)</td>.*?</div>'''
    results_i = re.findall(regix, text_i, re.S)
    # 将主讲人介绍信息进行拼接
#     print(len(results_i))
    txt_p,txt_le = results_i[0]
    regix_0 = '''>.*?<'''
    res_p = re.findall(regix_0, txt_p, re.S)
    res_p = ''.join(res_p)
    # 文本清洗
    intro_p = clean(res_p)
    intro_le = clean(txt_le)
    return intro_p, intro_le

# 爬取信息
def spider(url, headers):
    response = requests.get(url, headers = headers)
    response.encoding='utf-8'
    text = response.text
    regix = '''<li.*?>.*?<em class="arrow"></em>.*?<p class="room-cell"><a href="(.*?)" target="_blank" title="">(.*?):(.*?)</a>.*?<a href="" class="go_site" target="_blank"></a></p>.*?<label class="top">.*?时间：(.*?)</label>.*?<label class="top">.*?地点：(.*?)</label>.*?</li>'''
    results = re.findall(regix, text, re.S)
    for item in results:
        #判断是否为国外讲座
        pattern = re.compile('^[A-Za-z0-9.,:!?(--)_*"\' ]+$')
        if pattern.fullmatch(item[2]) == None:
            continue
        url_i = 'https://wise.xmu.edu.cn/'+ item[0][6:]
#         print(url_i)
        intro_p, intro_le = intro(url_i)
        dic = {
            '讲座时间': item[3],
            '讲座嘉宾': item[1],
            '嘉宾介绍': intro_p,
            '讲座题目': item[2],
            '讲座地点': item[4],
            '讲座简介': intro_le
        }
        file_data(dic)
        df = pd.DataFrame([dic])
        df.to_csv('学术讲座信息.csv', mode='a',encoding='utf-8', header=True)

# 爬虫进程
def main():
    # 爬取首页
    url_0 = 'https://wise.xmu.edu.cn/xsdt/xsjz/qb.htm'
    spider(url_0, headers)
    print('完成首页！')

    # 爬取后85页
    for i in range(1, 85):
        url_i = f'https://wise.xmu.edu.cn/xsdt/xsjz/qb/{86 - i}.htm'.format(i)
        spider(url_i, headers)
        print('完成第{}页！'.format(i + 1))

if __name__ == '__main__':
    main()
