# Eliot Abrams
# Food truck location choice


# Packages
from bs4 import BeautifulSoup
import os as os


# Folder locations
os.chdir('/Users/eliotabrams/Desktop/BBL')

# Download and organize data files in folder

# Import data
soup = BeautifulSoup(open("test.html"))

# Parse document
count = 0
days=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

for thing1 in soup.find_all('td'):
    if thing1.get('style') == 'width:13%':
        print '\n' + thing1.text
        count = 0

    else:
        print days[count]
        count +=1

    new_soup = BeautifulSoup(str(thing1), 'html.parser')

    for thing2 in new_soup.find_all('img'):
        print thing2.get('title')
        
        

