import redis

r = redis.Redis(host='localhost', port=6379)

# # 霸王别姬
# for i in range(493):
#     url = f'https://movie.douban.com/subject/1291546/reviews?start={i*20}'
#     r.lpush('start_urls', url)

# 阿甘正传
for i in range(350):
    url = f'https://movie.douban.com/subject/1292720/reviews?start={i*20}'
    r.lpush('start_urls', url)

# 泰坦尼克号
for i in range(430):
    url = f'https://movie.douban.com/subject/1292722/reviews?start={i*20}'
    r.lpush('start_urls', url)

# 千与千寻
for i in range(350):
    url = f'https://movie.douban.com/subject/1291561/reviews?start={i * 20}'
    r.lpush('start_urls', url)
#
# # 这个杀手不太冷
# for i in range(300):
#     url = f'https://movie.douban.com/subject/1295644/reviews?start={i * 20}'
#     r.lpush('start_urls', url)
#
# # 美丽人生
# for i in range(270):
#     url = f'https://movie.douban.com/subject/1292063/reviews?start={i * 20}'
#     r.lpush('start_urls', url)
#
# # 星际穿越
# for i in range(350):
#     url = f'https://movie.douban.com/subject/1889243/reviews?start={i * 20}'
#     r.lpush('start_urls', url)
#
# # 盗梦空间
# for i in range(345):
#     url = f'https://movie.douban.com/subject/3541415/reviews?start={i * 20}'
#     r.lpush('start_urls', url)
#
# # 楚门的世界
# for i in range(375):
#     url = f'https://movie.douban.com/subject/1292064/reviews?start={i * 20}'
#     r.lpush('start_urls', url)
#
# # 辛德勒的名单
# for i in range(150):
#     url = f'https://movie.douban.com/subject/1295124/reviews?start={i * 20}'
#     r.lpush('start_urls', url)

# # 忠犬八公的故事
# for i in range(220):
#     url = f'https://movie.douban.com/subject/3011091/reviews?start = {i * 20}'
#     r.lpush('start_urls', url)