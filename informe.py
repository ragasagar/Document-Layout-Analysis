from datetime import datetime
import time
import requests

try:
    
    r = requests.get('https://wirepusher.com/send?id=YkL9mpsRE&title=Program complete&message=CheckProgram&type=YourCustomType')
    # r = requests.get('https://wirepusher.com/send?id=hbBompXx6&title=Program Complete&message=CheckProgram&type=YourCustomType')
except Exception as e:
    print("Exception occur while sending email", e)