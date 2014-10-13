from unittest import TestCase
import kaggle_work as kwk

class KaggleWorkTest(TestCase):
    def test_get_circles(self):
        ego = 1
        all_friends_edges = {'1_2', '2_3', '1_4', '3_4'}
        raw_circles = \
        {
         '1' : {'C1', 'C2'},
         '2' : {'C2'},
         #'3' : {},
         '4' : {'C1'},
        }
        
        circles = kwk._get_circles(ego, all_friends_edges, raw_circles)
        print circles
        self.assertEquals(circles, {'1_2':'C2', '1_4':'C1'})
        