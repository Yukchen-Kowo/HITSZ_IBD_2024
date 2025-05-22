# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import pymongo

class DoubanPipeline:
    def __init__(self):
        super().__init__()
        self.client = pymongo.MongoClient()
        self.db = self.client['douban_movie_review']
        self.collection = self.db['review']
        self.count = 0

    def process_item(self, item, spider):
        if not self.collection.find_one(item):
            self.collection.insert_one(item)
            self.count += 1
            print(f'已抓取{self.count}条数据')
        return item

    def __del__(self):
        self.client.close()
