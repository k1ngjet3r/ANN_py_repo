from selenium import webdriver
import time
from datetime import datetime
import time
import matplotlib.pyplot as plt

driver = webdriver.Chrome('/Users/Jeter/Downloads/chromedriver')
driver.get('https://zlcsc.cyc.org.tw/')

time_points = []
amount = []


def data_logger():
    num = driver.find_element_by_id('gym_on').text
    time_point = str(datetime.now().time())[:-10]
    time_points.append(time_point)
    amount.append(num)
    print(time_point, num)


def diagram(x, y):
    plt.plot(x, y)
    plt.xlabel('Time Point')
    plt.ylabel('Number of People')
    plt.title(str(datetime.now().date()))
    plt.show


while True:
    data_logger()
    if datetime.now().hour == 22:
        print('times up')
        diagram(time_points, amount)
        break
    time.sleep(600)
