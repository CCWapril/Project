import pandas as pd
import requests
import re
import time
from bs4 import BeautifulSoup

# 遍历民宿
'''
爬取厦门市思明区与湖里区的民宿id，每次爬虫需要更新cookies与headers  
方法链接：https://zhuanlan.zhihu.com/p/518788491  
转码网站：https://curlconverter.com/
'''

cookies = {
    'cna': 'kpGUHCsJZF8BASQJiTRxUIaC',
    'xlly_s': '1',
    '_mw_us_time_': '1680854201152',
    't': '2848c50e5a9505231c065a8b0eaf3433',
    '_tb_token_': '73b3eb7e5e483',
    'cookie2': '2e69aee9a65dbdfe7e62a452642c9189',
    'cancelledSubSites': 'empty',
    'dnk': 'tb492736297',
    'uc1': 'cookie14=Uoe8iqMfdI6ipw%3D%3D&cookie15=VT5L2FSpMGV7TQ%3D%3D&pas=0&cookie21=V32FPkk%2Fgipm&cookie16=W5iHLLyFPlMGbLDwA%2BdvAGZqLg%3D%3D&existShop=false',
    'tracknick': 'tb492736297',
    'lid': 'tb492736297',
    '_l_g_': 'Ug%3D%3D',
    'unb': '4074078148',
    'cookie1': 'UNN8TkrrJK21%2FWWkSwTkpBAr6fx2EKgjb%2FnHnJGGZtg%3D',
    'login': 'true',
    'cookie17': 'VyyWsc1eRhPF8A%3D%3D',
    '_nk_': 'tb492736297',
    'sgcookie': 'E100EhChtnuBIIuPx2vf26otHe460JSQW4I7EOLZxM1e%2FsQTSX846j%2BlKWjMUUf4vcGtL8z7QUyLkVj93tLAhi2NoE6cuXIlBZZOm9NYLUXtiPo%3D',
    'sg': '78f',
    'csg': 'df22db8e',
    'x5sec': '7b22617365727665723b32223a223164396637643234366638666431643063316637393161366165396462303930434b6572763645474549724973732f6a344936485a686f4d4e4441334e4441334f4445304f4473784d4d7a533674722f2f2f2f2f2f77464141773d3d227d',
    'l': 'fBE3skZuNrTT-NrCBOfCFurza779jIRYSuPzaNbMi9fPOm195IuRW1iH7VTpCnhVFsW2R3ub1k7yBeYBq7VonxvtSFPEz4kmnmOk-Wf..',
    'tfstk': 'cRudBqGPOdvHTSJu7yKGUhjpsomdZpAuCgy5yqqVUK7HN5_Rif3mk6L2a724pCC..',
    'isg': 'BH19DQ7dxbP-SmFceLUM0KipjNl3GrFs9zZY5T_CulQDdp2oB2tfPnikIaowc8kk',
}

headers = {
    'authority': 'travelsearch.fliggy.com',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-language': 'zh-CN,zh;q=0.9',
    'cache-control': 'no-cache',
    # 'cookie': 'cna=kpGUHCsJZF8BASQJiTRxUIaC; xlly_s=1; _mw_us_time_=1680854201152; t=2848c50e5a9505231c065a8b0eaf3433; _tb_token_=73b3eb7e5e483; cookie2=2e69aee9a65dbdfe7e62a452642c9189; cancelledSubSites=empty; dnk=tb492736297; uc1=cookie14=Uoe8iqMfdI6ipw%3D%3D&cookie15=VT5L2FSpMGV7TQ%3D%3D&pas=0&cookie21=V32FPkk%2Fgipm&cookie16=W5iHLLyFPlMGbLDwA%2BdvAGZqLg%3D%3D&existShop=false; tracknick=tb492736297; lid=tb492736297; _l_g_=Ug%3D%3D; unb=4074078148; cookie1=UNN8TkrrJK21%2FWWkSwTkpBAr6fx2EKgjb%2FnHnJGGZtg%3D; login=true; cookie17=VyyWsc1eRhPF8A%3D%3D; _nk_=tb492736297; sgcookie=E100EhChtnuBIIuPx2vf26otHe460JSQW4I7EOLZxM1e%2FsQTSX846j%2BlKWjMUUf4vcGtL8z7QUyLkVj93tLAhi2NoE6cuXIlBZZOm9NYLUXtiPo%3D; sg=78f; csg=df22db8e; x5sec=7b22617365727665723b32223a223164396637643234366638666431643063316637393161366165396462303930434b6572763645474549724973732f6a344936485a686f4d4e4441334e4441334f4445304f4473784d4d7a533674722f2f2f2f2f2f77464141773d3d227d; l=fBE3skZuNrTT-NrCBOfCFurza779jIRYSuPzaNbMi9fPOm195IuRW1iH7VTpCnhVFsW2R3ub1k7yBeYBq7VonxvtSFPEz4kmnmOk-Wf..; tfstk=cRudBqGPOdvHTSJu7yKGUhjpsomdZpAuCgy5yqqVUK7HN5_Rif3mk6L2a724pCC..; isg=BH19DQ7dxbP-SmFceLUM0KipjNl3GrFs9zZY5T_CulQDdp2oB2tfPnikIaowc8kk',
    'pragma': 'no-cache',
    'referer': 'https://travelsearch.fliggy.com/index.htm?searchType=product&keyword=%E5%8E%A6%E9%97%A8%E6%80%9D%E6%98%8E%E5%8C%BA%E6%B0%91%E5%AE%BF',
    'sec-ch-ua': '"Chromium";v="110", "Not A(Brand";v="24", "Google Chrome";v="110"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
}

#爬取民宿信息
'''
同上，每次需要更新headers与cookies
'''
cookies2 = {
    'VISITED_HOTEL_TOKEN': '6944574c-0bb6-4165-87bd-138a776d1708',
    'cna': 'kpGUHCsJZF8BASQJiTRxUIaC',
    'lid': 'lover0504lyf',
    't': '2848c50e5a9505231c065a8b0eaf3433',
    'tracknick': 'lover0504lyf',
    '_tb_token_': '53e31e3d363a8',
    'cookie2': '10716f0c74cdc8c83af2a838e9de0756',
    'chanelStat': '"NA=="',
    'chanelStatExpire': '"2023-04-12 11:08:58"',
    'dnk': '%5Cu97A0%5Cu7AFA%5Cu9EE0_april',
    'cancelledSubSites': 'empty',
    'xlly_s': '1',
    'uc1': 'cookie14=Uoe8izHUzyWJ4A%3D%3D&existShop=false&pas=0&cookie15=UtASsssmOIJ0bQ%3D%3D&cookie16=VFC%2FuZ9az08KUQ56dCrZDlbNdA%3D%3D&cookie21=Vq8l%2BKCLjA%2Bl',
    '_l_g_': 'Ug%3D%3D',
    'unb': '2970606675',
    'cookie1': 'VTrlVzQ%2FCBbAFeOZkx%2BDXTmUfSOW6CQVDg6dYLfIln8%3D',
    'login': 'true',
    'cookie17': 'UUGlTiIS6g6crg%3D%3D',
    '_nk_': 'lover0504lyf',
    'sgcookie': 'E100G7HJPnVjiScwvoIP7%2FOFfqC76LMt4sbbmdj%2FTL4x9oRQaKcUzEwCa9LEYK4tUs0QMuuIdcvCtp0DBmyupn86C9a9SSeSn%2Bi5KrDJp2Kn0%2Bo%3D',
    'sg': 'f53',
    'csg': 'e613c165',
    'bx-cookie-test': '1',
    'x5sec': '7b226873703b32223a226463386131356462323262333034646238326336626330343034393131313639434f6e75797145474549335739374b646962474d52686f4d4d6a6b334d4459774e6a59334e5473304d4b44577849384751414d3d227d',
    'tfstk': 'cneOB9q2fwbgCkbmLACHuIhFEnyAZkSmRIuXH80Zy6rhhcpAikeuehBP0m0rWMC..',
    'l': 'fBMSxgteNrTa99HbBO5aPurza77tnIRb8sPzaNbMiIEGa1llTF6J-NCsu_TB7dtjgTfDoetrl5ZFzdHH-Va_WKOIs96-vRFvC0p9-bpU-L5..',
    'isg': 'BA0NWzkpVd6MvvHsZ1TvJdUoHCmH6kG8h2boNU-Sz6QTRiz4FzsxjgTUsNoghll0',
}

headers2 = {
    'authority': 'hotel.alitrip.com',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-language': 'zh-CN,zh;q=0.9',
    'cache-control': 'no-cache',
    # 'cookie': 'VISITED_HOTEL_TOKEN=6944574c-0bb6-4165-87bd-138a776d1708; cna=kpGUHCsJZF8BASQJiTRxUIaC; lid=lover0504lyf; t=2848c50e5a9505231c065a8b0eaf3433; tracknick=lover0504lyf; _tb_token_=53e31e3d363a8; cookie2=10716f0c74cdc8c83af2a838e9de0756; chanelStat="NA=="; chanelStatExpire="2023-04-12 11:08:58"; dnk=%5Cu97A0%5Cu7AFA%5Cu9EE0_april; cancelledSubSites=empty; xlly_s=1; uc1=cookie14=Uoe8izHUzyWJ4A%3D%3D&existShop=false&pas=0&cookie15=UtASsssmOIJ0bQ%3D%3D&cookie16=VFC%2FuZ9az08KUQ56dCrZDlbNdA%3D%3D&cookie21=Vq8l%2BKCLjA%2Bl; _l_g_=Ug%3D%3D; unb=2970606675; cookie1=VTrlVzQ%2FCBbAFeOZkx%2BDXTmUfSOW6CQVDg6dYLfIln8%3D; login=true; cookie17=UUGlTiIS6g6crg%3D%3D; _nk_=lover0504lyf; sgcookie=E100G7HJPnVjiScwvoIP7%2FOFfqC76LMt4sbbmdj%2FTL4x9oRQaKcUzEwCa9LEYK4tUs0QMuuIdcvCtp0DBmyupn86C9a9SSeSn%2Bi5KrDJp2Kn0%2Bo%3D; sg=f53; csg=e613c165; bx-cookie-test=1; x5sec=7b226873703b32223a226463386131356462323262333034646238326336626330343034393131313639434f6e75797145474549335739374b646962474d52686f4d4d6a6b334d4459774e6a59334e5473304d4b44577849384751414d3d227d; tfstk=cneOB9q2fwbgCkbmLACHuIhFEnyAZkSmRIuXH80Zy6rhhcpAikeuehBP0m0rWMC..; l=fBMSxgteNrTa99HbBO5aPurza77tnIRb8sPzaNbMiIEGa1llTF6J-NCsu_TB7dtjgTfDoetrl5ZFzdHH-Va_WKOIs96-vRFvC0p9-bpU-L5..; isg=BA0NWzkpVd6MvvHsZ1TvJdUoHCmH6kG8h2boNU-Sz6QTRiz4FzsxjgTUsNoghll0',
    'pragma': 'no-cache',
    'referer': 'https://hotel.alitrip.com/hotel_detail2.htm?shid=64515152&_output_charset=utf8',
    'sec-ch-ua': '"Chromium";v="110", "Not A(Brand";v="24", "Google Chrome";v="110"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
}

params = {
    '_output_charset': 'utf8',
}
# 定义爬虫对象
regix_list = ['<div class="base" meta-from="">.*?<h2>(.*?)<!--<em',# 民宿名称
              '<span class="row-subtitle" title="(.*?)" meta-level="(.*?)">', # 民宿评级（飞猪）
              '<p class="address">(.*?)</p>', # 民宿地址
              '<a href="#hotel-review" target="_self">(.*?)</a>', # 民宿分值与点评数
              '<span class="r-count">(.*?)</span>',# 好评数/差评数/有图评论数
              '<span class="pi-price" id="J_HotelPrice"><i>&yen;</i>(.*?)</span>', # 民宿平均价格
              '<p class="general-info">.*?<span>(.*?)</span>.*?<span>(.*?)</span>.*?</p>',# 房间平均面积
              '<div class="hotel-box hotel-desc" id="hotel-desc">.*?<div class="bd">(.*?)<br>', # 开业、装修、房间数
              '<div class="itemScore">(.*?)</div>',# 地理位置打分/清洁程度打分/服务体验打分/性价比打分
              '--<iframe src=".*?lat=(.*?)&lng=(.*?)&.*?"></iframe>--'
             ]
# dataframe列名
col = ['B&B','BB_Level','BB_Type','Location','lat&lng','Score','Comm_all','Comm_good',
       'Comm_bad','Comm_fig','Avg_price','Area_room','Open_year','Renov_year',
       'Num_room','Score_location','Score_clean','Score_serve','Score_price','Facility']

def geturl(params):
    response = requests.get('https://travelsearch.fliggy.com/index.htm', params=params, cookies=cookies, headers=headers)
    response.encoding='utf-8'
    text = response.text
    regix = '''<div class="product-left"><a href="(.*?)" target="_blank">'''
    url_results = re.findall(regix, text, re.S)
    return url_results

# BB:Bed & Breakfast
def InfoBB(url):
    BB_id = url[-8:]
    response = requests.get(url, params=params, cookies=cookies2, headers=headers2)
    response.encoding='utf-8'
    text = response.text
    info = pd.DataFrame(index=[BB_id],columns=col)
    try:
        # Name
        while True:
            info['B&B'] = re.findall(regix_list[0], text, re.S)
            # Level and Type
            res_1 = re.findall(regix_list[1], text, re.S)
            if len(res_1) == 0:
                break
            else:
                info['BB_Level'] = int(res_1[0][1][0])
                info['BB_Type'] = res_1[0][1][2:]
                break
        # Location
        while True:
            info['Location'] = re.findall(regix_list[2], text, re.S)
            info['lat&lng'] = re.findall(regix_list[9], text, re.S)
            break
        # Score and Comments
        while True:
            res_3 = re.findall(regix_list[3], text, re.S)
            if len(res_3) == 0:
                break
            if len(res_3) == 1:
                info['Comm_all'] = int(res_3[0])
                break
            else:
                info['Score'] = eval(res_3[0])
                info['Comm_all'] = int(res_3[1])
                break
        # Comments
        while True:
            res_4 = re.findall(regix_list[4], text, re.S)
            if len(res_4) == 0:
                break
            else:
                info['Comm_good'] = int(res_4[0][1:-1])
                info['Comm_bad'] = int(res_4[1][1:-1])
                info['Comm_fig'] = int(res_4[2][1:-1])
                break
        # Price
        while True:
            res_5 = re.findall(regix_list[5], text, re.S)[0]
            if res_5 == '--':
                break
            else:
                info['Avg_price'] = eval(res_5)
                break
        # Area of room
        res_6 = re.findall(regix_list[6], text, re.S)
        res_num = []
        for area in res_6:
            num = re.findall(r'\d+', area[1])
            res_num = res_num + num
        while True:
            if len(res_num) == 0:
                break
            else:
                info['Area_room'] = sum([eval(res) for res in res_num]) / len(res_num)
                break
        # Open year, renovate year and number of room
        while True:
            res_7 = re.findall(regix_list[7], text, re.S)
            find_open = res_7[0].find('年开业')
            find_renov = res_7[0].find('年装修')
            find_room = res_7[0].find('间房')
            if find_open == -1:
                break
            else:
                info['Open_year'] = 2023 - int(res_7[0][find_open - 4:find_open])
            if find_renov == -1:
                break
            else:
                info['Renov_year'] = 2023 - int(res_7[0][find_renov - 4:find_renov])
            if find_room == -1:
                break
            else:
                info['Num_room'] = int(res_7[0][find_room - 2:find_room])
            break
        # Score
        while True:
            res_8 = re.findall(regix_list[8], text, re.S)
            if len(res_8) == 0:
                break
            else:
                info['Score_location'] = eval(res_8[0])
                info['Score_clean'] = eval(res_8[1])
                info['Score_serve'] = eval(res_8[2])
                info['Score_price'] = eval(res_8[3])
                break
        # Facility
        soup = BeautifulSoup(text, 'html.parser')
        res_9 = soup.find('div',
                          attrs={"class": "hotel-box hotel-facility", "id": "hotel-facility"})  # 查找span class为red的字符串
        result = [span.get_text() for span in res_9.find_all('span')]
        info['Facility'] = [result]
    except ValueError as e:
        print('民宿{}检索失败'.format(BB_id))
        return info
    else:
        return info

if __name__ == '__main__':
    # 遍历民宿url
    url_list = []
    # 遍历思明区24页
#     for i in range(24):
    for i in range(24):
        params = {
            'searchType': 'product',
            'keyword': '厦门思明区民宿',
            'pagenum':i+1
        }
        url_1 = geturl(params)
        time.sleep(15)
        print('完成第{}页{}家民宿'.format(i+1,len(url_1)))
        url_list = url_list + url_1
    print(url_list)
    # 遍历湖里区8页
    for i in range(8):
        params = {
            'searchType': 'product',
            'keyword': '厦门湖里区民宿',
            'pagenum':i+1
        }
        url_2 = geturl(params)
        time.sleep(15)
        print('完成第{}页{}家民宿'.format(i+1,len(url_2)))
        url_list = url_list + url_2
    print(url_list)#正常应该是720个左右

    # 爬取民宿个体数据
    info_BB = pd.DataFrame(columns=col)
    # 遍历所有url
    for i in range(len(url_list)):
        BB = InfoBB(url_list[i])
        BB.to_csv('民宿信息.csv',sep=',',mode='a',index=True,header=True,encoding='utf8')
        info_BB = pd.concat([info_BB,BB])
        print('-----完成第{}家民宿，进度{:.3f}%-----'.format(i+1, (i+1)/len(url_list)*100))
        time.sleep(2.5)
