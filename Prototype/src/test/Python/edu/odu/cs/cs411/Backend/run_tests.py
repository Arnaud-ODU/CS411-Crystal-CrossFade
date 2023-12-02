#Driver File For Unit Tests For All Classes
#
#@author Joseph Wassell
#CS 411W Prof. Kennedy

from test_Note import *
from test_Track import *
from test_Song import *
from test_Song_changes import *

def test_All():
    #test_Note()
    #test_Track()
    #test_Song()
    test_song_changes()


def main():
    test_All()
    print('Funny Haha')
    
if __name__ == '__main__':
    main()