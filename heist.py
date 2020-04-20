# import os
# from selenium import webdriver
# import datetime
# import time

# option = webdriver.ChromeOptions()
# option.add_argument("--diaable-infobars")
# option.add_argument("--disable-notifications")
# browser = webdriver.Chrome(
#     '/Users/jeter/Downloads/chromedriver', chrome_options=option)

# browser.maximize_window()
# print('Opening Chrome...')
# browser.get('https://shopee.tw/product-i.127651399.2670033811')
# time.sleep(2)
# print('Opened the page successfully, trying to log-in to the website...')
# login_ele = browser.find_element_by_xpath(
#     "/html/body/div[1]/div/div[2]/div/div[2]/div/div[1]/div/div/div/div[1]/div/ul/li[5]")
# login_ele.click()
# time.sleep(2)

# print('entering the user infomation')

# username = browser.find_element_by_xpath(
#     '/html/body/div[2]/aside/div[1]/div/div/div/div[2]/div[1]/div[2]/div/input')
# password = browser.find_element_by_xpath(
#     '/html/body/div[2]/aside/div[1]/div/div/div/div[2]/div[1]/div[3]/div/input')
# submit = browser.find_element_by_xpath(
#     '/html/body/div[2]/aside/div[1]/div/div/div/div[2]/div[2]/button[2]')

# username.send_keys('0928590619')
# password.send_keys('xbox360')
# submit.click()

# print('account log-in successful')
# time.sleep(10)


# # print(browser.current_url)
# # target = browser.find_element_by_xpath(
# #     '/html/body/div[1]/div/div[2]/div/div[2]/div/div[14]/div/div/div/div/div[4]')
# # target.click()
# import numpy as np

# b = np.array([[-0.39295413],
#               [0.5131677],
#               [1.95352453],
#               [0.30982339],
#               [0.10793655],
#               [-0.08660033],
#               [1.50303502],
#               [-2.34255143],
#               [0.34313784],
#               [0.79029169],
#               [-0.16205514],
#               [-0.71151008],
#               [0.34866291],
#               [0.94227584],
#               [-0.43045028],
#               [0.69408095],
#               [-1.10605531],
#               [-0.54086024],
#               [-0.2718316],
#               [-0.94229782],
#               [1.43274685],
#               [-0.17001073],
#               [0.75693747],
#               [-1.12525984],
#               [1.19840558],
#               [1.01053424],
#               [0.15032947],
#               [-0.10604911],
#               [-0.36753689],
#               [-0.49584128]])
# print(b.shape)
# np.reshape(b, (len(b),))
# print(b.shape)


import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print((a - b))
