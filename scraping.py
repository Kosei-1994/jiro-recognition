import requests, sys, time, os
from bs4 import BeautifulSoup

cur_url = './pics/'
# スクレイピング対象の店舗名、URLをリスト内に保存
store_names = []
store_urls = []
max_page = 20
max_pic = 40

def download(store_url,max_page, max_pic, store_name, cur_url):
    if not os.path.isdir(cur_url+store_name):
        os.makedirs(cur_url+store_name)
    
    for num_page in range(max_page):
        soup = BeautifulSoup(requests.get(store_url+str(num_page+1)).content, 'lxml')
        print('{}ページ目のURL取得！'.format(num_page+1))
        time.sleep(10)
        images = []
        # URLのimgタグ取得
        links = soup.find_all('img')
        # print('{}ページ目のimgタグ取得！'.format(num_page+1))

        # 1ページの画像URL全部取得
        for i, link in enumerate(links):
            if i <max_pic:
                if link.get('src').endswith('.jpg'):
                    images.append(link.get('src'))
                elif link.get('src').endswith('.png'):
                    images.append(link.get('src'))
                print('{}個目の画像URL取得！'.format(i+1))
                time.sleep(2)
            else:
                break

        # iページ目の画像ダウンロード
        for j, target in enumerate(images):
            if j < max_pic:
                re = requests.get(target)
                with open(cur_url+store_name+'/'+target.split('/')[-1], 'wb') as f:
                    f.write(re.content)
                time.sleep(3)
            else:
                break
            print('{}枚目の画像ダウンロード完了！'.format(j+1))
    print(store_name+'のダウンロード終わり！！！')

def main():
    for store_name, store_url in zip(store_names, store_urls):
        download(store_url, max_page, max_pic, store_name, cur_url)

if __name__ == "__main__":
    main()