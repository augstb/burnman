import unittest

class BurnManTest(unittest.TestCase):
    def assertFloatEqual(self,a,b):
        self.assertAlmostEqual(a,b,delta=b*1e-5)

    def assertArraysAlmostEqual(self, a, b):
        self.assertEqual(len(a), len(b))
        for (i1, i2) in zip(a, b):
            self.assertFloatEqual(i1, i2)


class Huh(BurnManTest):
    def test(self):
        self.assertFloatEqual(5200.01, 5200.015)

if __name__ == '__main__':
    unittest.main()

        
