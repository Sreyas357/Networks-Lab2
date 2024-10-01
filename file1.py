# file1.py
from shared import shared_object


print("File 1: Accessing shared_object")
print(shared_object.name)

shared_object.name = "hi"