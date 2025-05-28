from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# 设置 WebDriver
driver = webdriver.Chrome()  # 确保已安装对应版本的chromedriver
driver.get("https://www.guessthepin.com/")  # 替换为实际的网页URL

try:
    # 尝试所有4位数组合 (0000-9999)
    for pin in range(0, 10000):
        try:
            # 尝试直接获取元素，不等待完全加载
            try:
                time.sleep(0.5)
                # 直接停止浏览器加载
                driver.execute_script("window.stop();")
                pin_input = driver.find_element(By.ID, "pin")
                submit_button = driver.find_element(By.XPATH, "//input[@type='submit' and @value='Guess']")
            except:
                # 如果元素未找到，刷新页面并继续
                driver.refresh()
                time.sleep(1)
                continue
            
            # 格式化PIN为4位数，前面补零
            pin_str = f"{pin:04d}"
            
            # 清除输入框并输入PIN
            pin_input.clear()
            pin_input.send_keys(pin_str)
            
            # 点击提交按钮
            submit_button.click()
            
            # 添加延迟以避免被检测为机器人
            time.sleep(0.5)  # 可以根据需要调整
            
            # 检查是否成功（根据实际页面调整）
            # 例如检查是否有错误消息或成功标志
            # if "success" in driver.page_source.lower():
            #     print(f"成功! PIN是: {pin_str}")
            #     break
            
            # 打印进度
            if pin % 100 == 0:  # 每100次打印一次进度
                print(f"尝试中... 当前PIN: {pin_str}")
                
        except Exception as e:
            print(f"尝试PIN {pin_str}时发生错误: {e}")
            # 如果出现错误，刷新页面并继续
            driver.refresh()
            time.sleep(1)
            continue

except KeyboardInterrupt:
    print("用户中断了程序")

finally:
    # 关闭浏览器
    driver.quit()
    print("程序结束")