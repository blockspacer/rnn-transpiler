import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import re
import time
import os
from pymongo import MongoClient
import pprint
from tqdm import tqdm
from dotenv import load_dotenv

# Parameters
machine = 'aws' # 'aws' or 'local'
num_problems = 439 # Number of problems on website

# Connect to database
if machine == 'aws':
    client = MongoClient()   
elif machine == 'local':
    client = MongoClient("mongodb://cjm715:password@3.17.141.166/py2cpp")
db = client.py2cpp
print(f'Number of python db documents: {db.python.count()}')
print(f'Number of C++ db documents: {db.cpp.count()}')

# Load website and credentials
load_dotenv()
EMAIL = os.environ.get('USERNAME')
PASSWORD = os.environ.get('PASSWORD')
WEBSITE = os.environ.get('WEBSITE')

# Language options
python_options = {
    'name' : 'Python 3',
    'suffix': '.py',
    'search_term': 'python3',
    'db_collection': db.python
}
cpp_options = {
    'name' : 'C++',
    'suffix': '.cpp',
    'search_term': 'cpp',
    'db_collection': db.cpp
}

# Setting up selenium web driver
if machine == 'aws':
    chromedriver = "/bin/chromedriver"
elif machine == 'local':
    chromedriver = "/Applications/chromedriver" 
os.environ["webdriver.chrome.driver"] = chromedriver
options = Options()
# Headless option

if machine == 'aws':
    options.add_argument("--headless")
    options.add_argument("--window-size=1920x1080")
    options.binary_location =  "/bin/headless-chromium"
     
# Helper function for sleeping
def sleeper(lower,higher):
    delay = lower + (higher-lower)*np.random.random()
    time.sleep(delay)

def open_website():
    # Open up website
    driver = webdriver.Chrome(chromedriver,options=options)
    driver.get(WEBSITE)
    sleeper(5,10)

    # Login
    username_field = driver.find_element_by_id('input-1')
    username_field.send_keys(EMAIL) 
    sleeper(5,10)
    pw_field = driver.find_element_by_id('input-2')
    pw_field.send_keys(PASSWORD) 
    sleeper(5,10)
    driver.find_elements_by_tag_name('button')[0].click()
    sleeper(5,10)
    driver.find_elements_by_tag_name('button')[0].click()
    sleeper(5,10)

    # Go to algorithms page
    driver.find_element_by_link_text('Algorithms').click()
    sleeper(5,10)
    
    return driver

# Function to go to a particular challenge problem by id
def go_to_leaderboard(req_challenge_id):
    assert req_challenge_id < num_problems
    driver.get('https://www.hackerrank.com/domains/algorithms')
    sleeper(5,10)

    # Scroll until you get to Challenge identified by req_challenge_id
    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        sleeper(3,4)

        # Break if number of challenges exceeds req_challenge_id
        challenge_list = driver.find_element_by_class_name('challenges-list')
        clist_items = challenge_list.find_elements_by_class_name('challenge-list-item')
        num_challenges_on_page = len(clist_items)
        if num_challenges_on_page > req_challenge_id:
            break

        sleeper(1,2)
        print(num_challenges_on_page)
    sleeper(5,10)
    
   
    # Get individual challenge data and click 
    challenge_list = driver.find_element_by_class_name('challenges-list')
    challenge_item = challenge_list.find_elements_by_class_name('challenge-list-item')[req_challenge_id]
    challenge_title = challenge_item.find_element_by_class_name('challengecard-title').text.split('\n')[0]
    challenge_difficulty = challenge_item.find_element_by_class_name('difficulty').text
    max_score_str = challenge_item.find_element_by_class_name('max-score').text
    max_score= float(re.findall('\d+', max_score_str)[0])
    success_rate_str = challenge_item.find_element_by_class_name('success-ratio').text
    success_rate = 0.01*float(re.findall('\\d+(?:\\.\\d+)?', success_rate_str)[0])
    print(f'challenge_title: {challenge_title}')
    print(f'challenge_difficulty: {challenge_difficulty}')
    print(f'max_score: {max_score}')
    print(f'success_rate: {success_rate}')

    challenge_item.click()
    sleeper(5,10)

    # Go to Leaderboard
    driver.find_element_by_link_text('Leaderboard').click()
    sleeper(5,10)
    reveal_button = driver.find_elements_by_tag_name('button')
    try:
        reveal_button[0].click()
        sleeper(5,10)
        driver.find_element_by_class_name('hr_primary-btn').click()
    except:
        print('solutions already revealed')

    sleeper(5,10) 
    return challenge_title,challenge_difficulty,max_score,success_rate

# Filter by language (used soon in two cells)

def filter_by_language(lang_options):
    sleeper(1,3)
    language_field = driver.find_elements_by_class_name('ac-input')[1]
    sleeper(1,3)
    for _ in range(len(language_field.get_attribute('value'))):
        language_field.send_keys(Keys.BACK_SPACE)
        sleeper(0.1,0.5)

    sleeper(1,3)
    language_field.send_keys(lang_options['search_term'])

    sleeper(1,3)
    language_field.send_keys(Keys.ENTER)
    sleeper(1,3)
    
    
# Collect solutions
min_page = 11
max_page = 19
solutions_per_page = 20 # Pick number less than 20. 20 is the number per page.


driver = open_website()

#Loop over problems
for problem_id in range(num_problems):
    c_title,c_difficulty,c_max_score,c_success_rate=go_to_leaderboard(problem_id)

    # Loop over languages
    for lang_options in [python_options, cpp_options]:
        print(f'Language: {lang_options["name"]} \n')
        filter_by_language(lang_options)

        # Loops through pages
        last_page = int(driver.find_element_by_class_name('last-page').text) - 1
        min_page_capped = min(min_page,last_page)
        max_page_capped = min(max_page,last_page) 
        for i in range(max_page_capped + 1):
            if i >= min_page_capped:
                
                # Loops through solutions on page
                solutions_per_page=len(driver.find_elements_by_class_name('table-row'))
                for j in range(solutions_per_page):

                    # Get meta data on solution such as rank, language, and score
                    print(j,solutions_per_page,len(driver.find_elements_by_class_name('table-row')))
                    
                    row = driver.find_elements_by_class_name('table-row')[j]
                    elementList = row.find_elements_by_class_name('table-row-column')
                    text_info = [ elem.text for elem in elementList ]
                    rank = int(text_info[1])
                    language = text_info[3]
                    score = float(text_info[4])

                    # Navigate to solution page and get code as a string
                    solution_links = driver.find_elements_by_link_text('View solution')
                    lin = solution_links[j]
                    url = lin.get_attribute('href')
                    driver.get(url)
                    sleeper(1.0,2.0)
                    b = driver.find_element_by_tag_name('body')
                    code = b.text

                    # Save to mongodb server
                    doc = {
                        'challenge_title': c_title,
                        'challenge_difficulty': c_difficulty,
                        'max_score': c_max_score,
                        'success_rate': c_success_rate,
                        'rank' : rank,
                        'language' : language,
                        'score': score,
                        'code' : code
                    }
        #             pp.pprint(doc)
                    try:
                        lang_options['db_collection'].insert_one(doc)
                    except:
                        pass
                    
                    # Navigate back to solutions list
                    driver.back()
                    sleeper(1.0,2.0)

            # Navigate to next page if it exists
            next_page_elem = driver.find_element_by_class_name('next-page')
            if 'disabled' in next_page_elem.get_attribute('class'):
                break
            else:
                next_page_elem.find_element_by_tag_name('a').click()
            sleeper(3,6) 