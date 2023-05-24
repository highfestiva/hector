import json
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait


cmd_filename = 'cmds/generic-actions.json'
logger = logging.getLogger(__name__)


def load_actions(filename=None):
    actions = json.load(open(filename or cmd_filename))
    actions['_driver'] = webdriver.Edge()
    # print(dir(actions['_driver']))
    return actions


def run(actions, action):
    exec_info = actions[action]
    driver = actions['_driver']
    if 'login_url' in exec_info:
        _login(driver, exec_info, actions=actions, action_name=action)
    elif 'url' in exec_info:
        _open_url(driver, exec_info, actions=actions, action_name=action)
    else:
        assert False, f'command config of {action} erroneous'


def _login(driver, exec_info, actions, action_name):
    url = exec_info['login_url']
    actions['_base_url'] = url.rsplit('/', 1)[0]
    usel,username = exec_info['username'].split(':', 1)
    psel,password = exec_info['password'].split(':', 1)
    ssel = exec_info['submit']
    logger.debug('%s: loading login %s', action_name, url)
    driver.get(url)
    WebDriverWait(driver, 10).until(lambda driver: driver.find_element(By.CSS_SELECTOR, ssel))
    driver.find_element(By.CSS_SELECTOR, usel).send_keys(username)
    driver.find_element(By.CSS_SELECTOR, psel).send_keys(password)
    driver.find_element(By.CSS_SELECTOR, ssel).click()


def _open_url(driver, exec_info, actions, action_name):
    url = exec_info['url']
    if '://' not in url:
        url = actions.get('_base_url','') + url
    logger.debug('%s: loading %s', action_name, url)
    driver.get(url)
