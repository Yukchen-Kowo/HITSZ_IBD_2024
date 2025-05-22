import time
import scrapy
from scrapy_redis.spiders import RedisSpider
from bs4 import BeautifulSoup
import queue
import redis
from threading import Thread
queue = queue.Queue(500)


class TheshawshankredemptionSpider(RedisSpider):
    name = "TheShawshankRedemption"
    allowed_domains = ["movie.douban.com"]
    # start_urls = ["https://movie.douban.com/subject/1292052/reviews"]
    redis_key = 'start_urls'

    def __init__(self):
        super().__init__()
        self.thread = Thread(target=self.return_url)
        self.thread.start()

    def parse(self, response):
        urls = response.xpath('//div[@class="main-bd"]/h2/a/@href').getall()
        if not urls:
            queue.put(response.url)
            print('已返回到redis库，等待重新请求')

        for url in urls:
            yield scrapy.Request(url=url, headers={'Referer': response.url}, callback=self.parse_detail)

        # next_page = response.xpath('//span[@class="next"]/link/@href').get()
        # if next_page is not None:
        #     yield response.follow('/subject/1292052/reviews' + next_page, callback=self.parse)

    def parse_detail(self, response):
        content = response.xpath('//div[@class="review-content clearfix"]').getall()
        content = ''.join(content)
        content = content.replace('\n', '').replace('\xa0', '').replace(' ', '')
        text = BeautifulSoup(content, 'html.parser').get_text()
        yield {'text': text}


    def return_url(self):
        r = redis.Redis(host='localhost', port=6379)
        while True:
            while queue.not_empty:
                url = queue.get()
                r.rpush('start_urls', url)
                time.sleep(0.1)
            time.sleep(1)

    def __del__(self):
        self.thread.join()