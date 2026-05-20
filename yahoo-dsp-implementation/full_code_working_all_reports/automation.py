# # -*- coding: utf-8 -*-
# """
# Created on Sun Mar 21 01:06:53 2021
 
# @author: VICKY JUNGHARE
# """
# #importing required python packages
# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.common.action_chains import ActionChains
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.chrome.options import Options
# import time
# import random
 
# username= "luffysurpasskaido" #Instagram Username
# password= "Pikachuluffy@1995" #Instagram Password
 


# chrome_options = Options()
# chrome_options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# driver = webdriver.Chrome(
#     service=Service(ChromeDriverManager().install()),
#     options=chrome_options
# )
# driver.get("https://www.instagram.com/")
# enter_username = WebDriverWait(driver,20).until(EC.presence_of_element_located((By.NAME, 'username')))
# enter_username.send_keys(username)
   
# enter_password = WebDriverWait(driver,20).until(EC.presence_of_element_located((By.NAME, 'password')))
# enter_password.send_keys(password)
# enter_password.send_keys(Keys.RETURN)
# time.sleep(random.randint(1,4))
 
# try:
#     not_now = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'mt3GC')))
#     a = not_now.find_elements(By.TAG_NAME, "button")[1]
#     actions = ActionChains(driver)
#     actions.click(a)
#     actions.perform()
# except Exception as e:
#     print("No 'Not Now' button found or error occurred:", e)
#     pass
 
# finally:
#     for i in range(500):  # Number of views
#         driver.get("https://www.instagram.com/reel/DYcaa6yInbZ/")  # Reel link
#         time.sleep(random.randint(1, 2))
        



# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.common.action_chains import ActionChains
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager
# import time
# import random

# username= "luffysurpasskaido" #Instagram Username
# password= "Pikachuluffy@1995" #Instagram Password

# # Add options to bypass bot detection
# options = Options()
# options.add_argument("--window-size=1920,1080")
# options.add_argument("--disable-blink-features=AutomationControlled")
# options.add_experimental_option("excludeSwitches", ["enable-automation"])
# options.add_experimental_option("useAutomationExtension", False)
# options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# # Hide webdriver property from JavaScript
# driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
#     "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
# })

# driver.get("https://www.instagram.com/")
# time.sleep(random.randint(3, 5))  # Wait longer for page to fully load

# # enter_username = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.NAME, 'username')))
# # enter_username.send_keys(username)

# # enter_password = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.NAME, 'password')))
# # enter_password.send_keys(password)
# # enter_password.send_keys(Keys.RETURN)
# # time.sleep(random.randint(1, 4))

# # Replace this:
# enter_username = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.NAME, 'username')))
# enter_username.send_keys(username)

# enter_password = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.NAME, 'password')))
# enter_password.send_keys(password)
# enter_password.send_keys(Keys.RETURN)

# # With this:
# time.sleep(3)  # Let page fully settle

# # Try multiple selectors for username
# enter_username = WebDriverWait(driver, 20).until(
#     EC.element_to_be_clickable((By.CSS_SELECTOR, 'input[name="username"], input[aria-label="Mobile number, username or email"]'))
# )
# enter_username.click()
# time.sleep(1)
# # Type slowly like a human
# for char in username:
#     enter_username.send_keys(char)
#     time.sleep(random.uniform(0.05, 0.15))

# time.sleep(1)

# enter_password = WebDriverWait(driver, 20).until(
#     EC.element_to_be_clickable((By.CSS_SELECTOR, 'input[name="password"], input[aria-label="Password"]'))
# )
# enter_password.click()
# time.sleep(1)
# for char in password:
#     enter_password.send_keys(char)
#     time.sleep(random.uniform(0.05, 0.15))

# time.sleep(1)
# enter_password.send_keys(Keys.RETURN)

# try:
#     not_now = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'mt3GC')))
#     a = not_now.find_elements(By.TAG_NAME, "button")[1]
#     actions = ActionChains(driver)
#     actions.click(a)
#     actions.perform()
# except Exception as e:
#     print("No 'Not Now' button found or error occurred:", e)
#     pass

# finally:
#     for i in range(500):
#         driver.get("https://www.instagram.com/reel/DYcaa6yInbZ/")  # Reel link
#         time.sleep(random.randint(1, 2))
 
 
 
 



# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.common.action_chains import ActionChains
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager
# import time
# import random

# username= "luffysurpasskaido" #Instagram Username
# password= "Pikachuluffy@1995" #Instagram Password

# options = Options()
# options.add_argument("--window-size=1920,1080")
# options.add_argument("--disable-blink-features=AutomationControlled")
# options.add_experimental_option("excludeSwitches", ["enable-automation"])
# options.add_experimental_option("useAutomationExtension", False)
# options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
#     "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
# })

# driver.get("https://www.instagram.com/")
# time.sleep(5)



# # Debug: print all input fields found on the page
# inputs = driver.find_elements(By.TAG_NAME, "input")
# print(f"Found {len(inputs)} input fields:")
# for inp in inputs:
#     print(f"  name='{inp.get_attribute('name')}' | type='{inp.get_attribute('type')}' | aria-label='{inp.get_attribute('aria-label')}'")



# # ---- Login ----
# enter_username = WebDriverWait(driver, 20).until(
#     EC.element_to_be_clickable((By.CSS_SELECTOR, 'input[name="username"]'))
# )
# enter_username.click()
# time.sleep(1)
# for char in username:
#     enter_username.send_keys(char)
#     time.sleep(random.uniform(0.05, 0.15))

# time.sleep(1)

# enter_password = WebDriverWait(driver, 20).until(
#     EC.element_to_be_clickable((By.CSS_SELECTOR, 'input[name="password"]'))
# )
# enter_password.click()
# time.sleep(1)
# for char in password:
#     enter_password.send_keys(char)
#     time.sleep(random.uniform(0.05, 0.15))

# time.sleep(1)
# enter_password.send_keys(Keys.RETURN)
# time.sleep(random.randint(3, 5))

# # ---- Dismiss popups ----
# try:
#     not_now = WebDriverWait(driver, 10).until(
#         EC.presence_of_element_located((By.XPATH, "//button[text()='Not Now']"))
#     )
#     not_now.click()
# except Exception as e:
#     print("No 'Not Now' button found:", e)

# # ---- Main loop ----
# for i in range(500):
#     driver.get("https://www.instagram.com/reel/DYcaa6yInbZ/")  # Reel link
#     time.sleep(random.randint(1, 2))
    
    
    
    
    
    
    
    
    

# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.common.action_chains import ActionChains
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager
# import time
# import random

# # ---- Credentials ----
# username= "luffysurpasskaido" #Instagram Username
# password= "Pikachuluffy@1995" #Instagram Password

# # ---- Browser Setup ----
# options = Options()
# options.add_argument("--window-size=1920,1080")
# options.add_argument("--disable-blink-features=AutomationControlled")
# options.add_experimental_option("excludeSwitches", ["enable-automation"])
# options.add_experimental_option("useAutomationExtension", False)
# options.add_argument(
#     "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
#     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
# )

# driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# # Hide webdriver fingerprint
# driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
#     "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
# })

# # ---- Open Instagram ----
# driver.get("https://www.instagram.com/")
# time.sleep(5)

# # ---- Login ----
# try:
#     enter_username = WebDriverWait(driver, 20).until(
#         EC.element_to_be_clickable((By.NAME, 'email'))
#     )
#     enter_username.click()
#     time.sleep(1)
#     for char in username:
#         enter_username.send_keys(char)
#         time.sleep(random.uniform(0.05, 0.15))

#     time.sleep(1)

#     enter_password = WebDriverWait(driver, 20).until(
#         EC.element_to_be_clickable((By.NAME, 'pass'))
#     )
#     enter_password.click()
#     time.sleep(1)
#     for char in password:
#         enter_password.send_keys(char)
#         time.sleep(random.uniform(0.05, 0.15))

#     time.sleep(1)
#     enter_password.send_keys(Keys.RETURN)
#     print("Login submitted successfully")
#     time.sleep(random.randint(4, 6))

# except Exception as e:
#     print(f"Login failed: {e}")
#     driver.quit()
#     exit()

# # ---- Dismiss 'Save Login Info' Popup ----
# try:
#     not_now = WebDriverWait(driver, 10).until(
#         EC.element_to_be_clickable((By.XPATH, "//button[text()='Not Now']"))
#     )
#     not_now.click()
#     print("Dismissed 'Save Login Info' popup")
#     time.sleep(2)
# except Exception as e:
#     print(f"No 'Save Login Info' popup: {e}")

# # ---- Dismiss 'Turn on Notifications' Popup ----
# try:
#     not_now2 = WebDriverWait(driver, 10).until(
#         EC.element_to_be_clickable((By.XPATH, "//button[text()='Not Now']"))
#     )
#     not_now2.click()
#     print("Dismissed 'Notifications' popup")
#     time.sleep(2)
# except Exception as e:
#     print(f"No 'Notifications' popup: {e}")

# # ---- Main Loop ----
# reel_url = "https://www.instagram.com/reel/DYcaa6yInbZ/"  # Reel link
# total_views = 500

# print(f"Starting {total_views} reel visits...")
# for i in range(total_views):
#     try:
#         driver.get(reel_url)
#         time.sleep(random.randint(1, 2))
#         print(f"Visit {i+1}/{total_views} done")
#     except Exception as e:
#         print(f"Error on visit {i+1}: {e}")
#         continue

# print("Done!")
# driver.quit()




# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.support.ui import WebDriverWait, Select
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager
# import time
# import random
# import traceback
# import os
# import undetected_chromedriver as uc
#   # Instead of webdriver.Chrome(...)

# # ---- Credentials ----
# USERNAME = "luffysurpasskaido" #Instagram Username
# PASSWORD = "Pikachuluffy@1995" #Instagram Password
# REEL_URL = "https://www.instagram.com/reel/DYcaa6yInbZ"
# TOTAL_CYCLES = 50  # Start small for testing

# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.common.action_chains import ActionChains
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager
# import time
# import random
# import traceback

# # ---- Credentials ----
# USERNAME = "luffysurpasskaido" #Instagram Username
# PASSWORD = "Pikachuluffy@1995" #Instagram Password
# REEL_URL = "https://www.instagram.com/reel/DYcaa6yInbZ"
# TARGET_CYCLES = 100    # Number of view simulations (start small!)

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import random
import traceback

# ---- Credentials ----

USERNAME = "luffysurpasskaido" #Instagram Username
PASSWORD = "Pikachuluffy@1995" #Instagram Password

# ---- Your 3 URLs (mix of post + reels) ----
URLS = [
    "https://www.instagram.com/p/DYdnJNYI7FP/",      # Post
    "https://www.instagram.com/reel/DYcaa6yInbZ/",   # Reel
    "https://www.instagram.com/reel/DYcaArDo4wj/"    # Reel
]

CYCLES = 30  # Start small - test first

# ---- Browser Setup ----
options = Options()
options.add_argument("--window-size=1920,1080")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
options.add_experimental_option("useAutomationExtension", False)
options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Anti-detection JS
driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
    "source": """
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
        Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
    """
})

def human_type(element, text):
    for char in text:
        element.send_keys(char)
        time.sleep(random.uniform(0.03, 0.15))

def wait_for_media(driver, timeout=20):
    """Wait for video OR image to load (handles both /p/ and /reel/)"""
    wait = WebDriverWait(driver, timeout)
    try:
        # Try video first (reels)
        video = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'video')))
        driver.execute_script("return arguments[0].readyState >= 2", video)
        return video, 'video'
    except:
        try:
            # Fallback to image (posts)
            img = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'img[xpath*="Media"]')))
            return img, 'image'
        except:
            return None, None

def simulate_view(driver, url):
    """
    Simulate legitimate view behavior:
    - Load URL
    - Wait for media
    - Keep media in viewport for 5-10 seconds
    - Scroll naturally between items
    - Add mouse movement
    """
    try:
        driver.get(url)
        time.sleep(random.uniform(3, 6))
        
        media, media_type = wait_for_media(driver)
        if not media:
            print(f"⚠ No media found on {url}")
            return False
        
        # If video: ensure it's playing and visible
        if media_type == 'video':
            driver.execute_script("""
                let v = arguments[0];
                v.muted = true;
                v.scrollIntoView({behavior: 'smooth', block: 'center'});
                if (v.paused) v.play().catch(e=>{});
            """, media)
            # Watch time: 5-10 seconds (Instagram's threshold is ~3s)
            time.sleep(random.uniform(5, 10))
        else:
            # For posts: scroll into view and pause (image posts count "views" differently)
            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'})", media)
            time.sleep(random.uniform(3, 7))
        
        # Simulate browsing: scroll down slightly, then back
        driver.execute_script(f"window.scrollBy(0, {random.randint(300, 700)});")
        time.sleep(random.uniform(2, 4))
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(random.uniform(1, 3))
        
        # Mouse movement simulation
        actions = ActionChains(driver)
        for _ in range(random.randint(3, 7)):
            actions.move_by_offset(random.randint(-30, 30), random.randint(-20, 20))
            time.sleep(random.uniform(0.2, 0.6))
        actions.perform()
        
        return True
        
    except Exception as e:
        print(f"⚠ Error on {url}: {e}")
        traceback.print_exc()
        return False

# ---- Login (Your Working Flow) ----
print("🔐 Logging in...")
driver.get("https://www.instagram.com/")
time.sleep(5)

try:
    user_field = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.NAME, 'email')))
    user_field.click()
    time.sleep(0.5)
    human_type(user_field, USERNAME)
    time.sleep(random.uniform(0.5, 1.5))
    
    pass_field = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.NAME, 'pass')))
    pass_field.click()
    time.sleep(0.5)
    human_type(pass_field, PASSWORD)
    time.sleep(random.uniform(0.5, 1.5))
    
    pass_field.send_keys(Keys.RETURN)
    print("✓ Login submitted")
    time.sleep(random.randint(6, 10))
except Exception as e:
    print(f"✗ Login failed: {e}")
    driver.save_screenshot("login_fail.png")
    driver.quit()
    exit(1)

# Dismiss popups
for _ in range(2):
    try:
        btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//button[text()='Not Now']"))
        )
        btn.click()
        time.sleep(1)
    except:
        pass

if "login" in driver.current_url.lower():
    print("✗ Login failed - still on login page")
    driver.quit()
    exit(1)

print("✓ Login successful")

# ---- Main Loop: Cycle Through All 3 URLs ----
print(f"🎬 Starting {CYCLES} cycles through {len(URLS)} URLs...")

for cycle in range(CYCLES):
    print(f"\n🔄 Cycle {cycle+1}/{CYCLES}")
    
    for idx, url in enumerate(URLS, 1):
        try:
            print(f"  ▶ Processing [{idx}/{len(URLS)}]: {url.split('/')[-2]}")
            
            if simulate_view(driver, url):
                print(f"    ✓ View simulated")
            else:
                print(f"    ⚠ Failed to simulate")
            
            # Short delay between URLs in same cycle
            if idx < len(URLS):
                time.sleep(random.randint(3, 8))
                
        except Exception as e:
            print(f"    ✗ Error: {e}")
            continue
    
    # Longer delay between full cycles (critical for avoiding detection)
    if cycle < CYCLES - 1:
        delay = random.randint(25, 45)
        print(f"  ⏱ Waiting {delay}s before next cycle...")
        time.sleep(delay)

print(f"\n✅ Completed {CYCLES} cycles")
print("🔒 Closing browser...")
driver.quit()