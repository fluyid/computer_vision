1. Open this in pycharm
2. Press Alt + F12 to open terminal (which is the third last icon from the bottom left)
3. Then run this command: python -m pip install -r requirements.txt
4. If that doesn't work try this one: py -m pip install -r requirements.txt
5. If you're having some sort of permission issues run this (copy and paste the whole thing):
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
6. Make sure you're in screenshot_generator.py and press shift + F10 (or that little play button on top) to run the file
7. The program should start capturing in 60 seconds ("I set this as a value because of loading screen and stuff")
8. If you want to change that value from 60, just go to function that looks like "def is_after_first_minute(self)" and 
change the value from 60 to your desired value
