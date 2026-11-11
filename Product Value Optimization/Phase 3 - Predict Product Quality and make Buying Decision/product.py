class Product:
    """A product with a feature array, a value and a price."""

    def __init__(self, features, value, price):
        self.features = features
        self.value = value
        self.price = price

    def __repr__(self):
        return f"Product[v={self.value}, p={self.price}]"

