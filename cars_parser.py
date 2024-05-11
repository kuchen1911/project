import os
import re
import time
import httpx
import logging

from uuid import uuid4
from bs4 import BeautifulSoup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARSING_DIR = os.path.join(ROOT_DIR, 'cars2')

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9,ru-RU;q=0.8,ru;q=0.7,ar-JO;q=0.6,ar;q=0.5,zh-CN;q=0.4,zh-TW;q=0.3,zh;q=0.2",
    "Cache-Control": "max-age=0",
    "Priority": "u=0, i",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

def sanitize_filename(filename):
    sanitized_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return sanitized_filename

class CarsParser:
    def __init__(self):
        self.images_parsed = 0 # Количество спаршенных фото в текущей категории
        self.total_parsed = 0  # Общее кол-во спаршенных фото
    
        self.client = httpx.Client()
    
        # Список категорий
        self.categories = [
            # 'van',
            'convertible',
            'coupe',
            'hatchback',
            'minivan',
            'pickup_truck',
            'suv',
            'sedan',
            'wagon',
        ]
        
        self.init_logger()

    def init_logger(self):
        '''
            Инициализируем логгер
        '''
        
        self.logger = logging.getLogger('CarsParserLogger')
        self.logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        self.logger.addHandler(ch)
    
    def start(self, pages=10):
        '''
            Метод начала парсинга
            
            @param pages - Кол-во страниц для парсинга
        '''
        
        self.total_parsed = 0
        
        # Создаем папку если не существует
        if not os.path.exists(PARSING_DIR):
            os.mkdir(PARSING_DIR)
        
        # Для каждой категории парсим заданное количество страниц с машинами
        for category in self.categories:
            self.logger.info(f'Parsing category: {category}')
            
            folder = os.path.join(PARSING_DIR, category)
            
            if not os.path.exists(folder):
                os.mkdir(folder)
            
            self.images_parsed = 0
            for i in range(pages):
                self.parse_page(i + 1, category)
    
    def parse_page(self, page: int, category: str):
        # Формируем ссылку
        link = f'https://www.cars.com/shopping/results/?body_style_slugs[]={category}&page={page}&dealer_id=&keyword=&list_price_max=&list_price_min=&makes[]=&maximum_distance=all&mileage_max=&monthly_payment=&page_size=100&sort=best_match_desc&stock_type=all&year_max=&year_min=&zip='

        # Отправляем запрос
        resp = self.client.get(link, headers=HEADERS)
        if resp.status_code != 200:
            self.logger.error(f"ERROR: {resp.status_code} TEXT: {resp.text}")
            return
        
        soup = BeautifulSoup(resp.content, 'lxml')
        
        # Получаем карточки машин из html кода
        cards = soup.select('.vehicle-card-main')
        
        self.logger.info(f'Got {len(cards)} cars on page: {page}')
        
        # Из каждой карточки получаем фото и название машины и сохраняем файл
        for card in cards:        
            title = card.select_one('.title').text
            title = sanitize_filename(title)
            
            images = card.select('.gallery-wrap .image-wrap img')[:3]

            uuid = str(uuid4())[:8]
            for idx, image in enumerate(images):
                
                link = image.attrs.get('data-src', '')
                if not link:
                    continue
                
                idx_str = str(idx + 1).zfill(5)
                filename = f'{idx_str}_{title}_{uuid}.jpg'
                output_file = os.path.join(PARSING_DIR, category, filename)
                
                self.download_image(output_file, link)
                
                self.images_parsed += 1
                self.total_parsed += 1
            
            time.sleep(0.1)

    def download_image(self, output: str, link: str, retries=5):
        # Пытаемся скачать файл заданное количество раз
        for _ in range(retries):
            try:
                with httpx.stream('GET', link, timeout=60) as stream:
                    with open(output, 'wb') as f:
                        for chunk in stream.iter_bytes():
                            f.write(chunk)
                return
            
            except Exception as e:
                self.logger.error(f"Download failed: {e}. Retrying...")
                time.sleep(1)
                
        self.logger.error("Download failed after multiple retries")

if __name__ == '__main__':
    parser = CarsParser()
    parser.start()