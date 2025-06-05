from api_builder import APIBuilder

class AppFactory:
    def __init__(self):
        self.app = APIBuilder().app

app = AppFactory().app
