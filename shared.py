# shared_module.py
class MyClass:
    def __init__(self):
        print("MyClass object created")
        self.name = "SharedObject"

# This object is created when the module is first imported
shared_object = MyClass()
