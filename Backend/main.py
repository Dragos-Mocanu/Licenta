from api_builder import APIBuilder

class Main:
    def __init__(self):
        self.app = APIBuilder().app

app = Main().app