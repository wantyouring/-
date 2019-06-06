from selenium import webdriver

driver = webdriver.Chrome('/Users/wantyouring/PycharmProjects/untitled/venv/'
                          '연습코드/network/chromedriver.exe')
driver.implicitly_wait(3)
driver.get('https://nid.naver.com/nidlogin.login')

driver.find_element_by_name('id').send_keys('pwc99')
driver.find_element_by_name('pw').send_keys('1111')
# 소스검사에서 onclick속성들 찾기.
driver.find_element_by_xpath('//form[@id="frmNIDLogin"]/fieldset[@class="login_form"]/input').click()
#//element[@속성]/ ... /최종 찾을 element