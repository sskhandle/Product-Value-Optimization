'''
The product class
'''

class Product(object):
    '''
    Represents a single product.
    Each product has a price, a value, and a feature array
    '''

    def __init__(self, features, value, price):
        self.features = features
        self.value = value
        self.price = price
    
    def __repr__(self):
        return "Product[v=" + str(self.value) + ", p=" + str(self.price) +"]"
        