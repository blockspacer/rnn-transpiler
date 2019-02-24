# !pip install selenium
# !pip install pymongo
# !pip install tqdm
# !pip install python-dotenv

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
from tqdm import tqdm_notebook, tqdm

# Number of problems on website
num_problems = 439

# pprint
pp = pprint.PrettyPrinter(indent=4)

# Connect to database
client = MongoClient()
db = client.py2cpp

# Load website and credentials
from dotenv import load_dotenv
load_dotenv()
EMAIL = os.environ.get('USERNAME')
PASSWORD = os.environ.get('PASSWORD')
WEBSITE = os.environ.get('WEBSITE')
print(WEBSITE)

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

# Helper function for sleeping
def sleeper(lower,higher):
    delay = lower + (higher-lower)*np.random.random()
    time.sleep(delay)


chromedriver = "/bin/chromedriver" # path to the chromedriver executable
os.environ["webdriver.chrome.driver"] = chromedriver
options = Options()
# Headless option
options.add_argument("--headless")
options.add_argument("--window-size=1920x1080")
options.binary_location =  "/bin/headless-chromium"

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
max_page_number = 20
solutions_per_page = 5 # Pick number less than 20. 20 is the number per page.

#Loop over problems
for problem_id in range(num_problems):
    c_title,c_difficulty,c_max_score,c_success_rate=go_to_leaderboard(problem_id)

    # Loop over languages
    for lang_options in [python_options, cpp_options]:
        print(f'Language: {lang_options["name"]} \n')
        filter_by_language(lang_options)

        # Loops through pages
        final_page_num = int(driver.find_element_by_class_name('last-page').text)
        last_page = min(max_page_number,final_page_num) 
        for i in tqdm(range(last_page)):

            # Loops through solutions on page
            for j in range(solutions_per_page):
                
                # Get meta data on solution such as rank, language, and score
                row = driver.find_elements_by_class_name('table-row')[j]
                sleeper(0,1)
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
                lang_options['db_collection'].insert_one(doc)
                sleeper(1,2)
                # Navigate back to solutions list
                driver.back()

                sleeper(1,2)


            # Navigate to next page if it exists
            sleeper(2,4)   
            next_page_elem = driver.find_element_by_class_name('next-page')
            if 'disabled' in next_page_elem.get_attribute('class'):
                break
            else:
                next_page_elem.click()
            sleeper(3,6) 
