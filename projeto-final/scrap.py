import scrapy

class BlogSpider(scrapy.Spider):
    name = 'blogspider'
    start_urls = ['http://guitarpatches.com/patches.php?unit=G3']

    def parse(self, response):
        for link in response.css('.lists tbody a'):
            yield {
                'title': link.css('a::text').extract_first(),
                'link': link.css('a::attr(href)').extract_first(),
                'index': link.css('a::attr(href)').extract_first().split('=')[-1],
            }

        for next_page in response.xpath('//a[contains(text(), "next")]'):
            yield response.follow(next_page, self.parse)


from scrapy.crawler import CrawlerProcess

process = CrawlerProcess({
    'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
    'FEED_FORMAT': 'json',
    'FEED_URI': 'data/pedalboards-list.json',
})


process.crawl(BlogSpider)
process.start()
process.stop()

