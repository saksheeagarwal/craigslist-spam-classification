import scrapy
from scrapy import Request


class CraigslistSpider(scrapy.Spider):
    name = 'data'
    allowed_domains = ['craigslist.org']
    start_urls = ['https://chicago.craigslist.org/']
    # scrap ad category from the current layer
    def parse(self, response):
        cat = response.xpath('//*[@id="sss"]//*[@class="cats"]//li')
        for c in cat:
            category = c.xpath('a/span/text()').extract()
            lower_rel_url = c.xpath('a/@href').extract_first()
            lower_url = response.urljoin(lower_rel_url)
            yield Request(lower_url, callback=self.parse_lower, 
                meta={'Category': category})
    # scrap ad title from the next layer 
    def parse_lower(self, response): 
        ads = response.xpath('//p[@class="result-info"]')
        for ad in ads:  # extract each ad info
            title = ad.xpath('a/text()').extract()
            response.meta['Title'] = title
            lower_rel_url = ad.xpath('a/@href').extract_first()
            lower_rul = response.urljoin(lower_rel_url)
            yield Request(lower_rul, callback=self.parse_sec_lower, 
                meta={'Title':title})
        #  scrape from multiple pages
        next_rel_url = response.xpath('//a[@class="button next"]/@href').get()
        next_url = response.urljoin(next_rel_url)
        yield Request(next_url, callback=self.parse_lower)
    # scrap ad description from the second layer
    def parse_sec_lower(self, response):
        category = response.xpath('//*[@class="crumb category"]/p/a/text()').extract()
        #category = category[0][:category[0].find(' -')]
        text = "".join(line for line in response.xpath('//*[@id="postingbody"]/text()').getall()) 
        text = text.lstrip()
        response.meta['Description'] = text
        response.meta['Category'] = category
        yield response.meta

