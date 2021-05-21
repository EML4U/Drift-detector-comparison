# https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED
# If this variable is not set or set to random, a random value is used to seed the hashes of str and bytes objects.
# If PYTHONHASHSEED is set to an integer value, it is used as a fixed seed for generating the hash() of the types covered by the hash randomization.
# The integer must be a decimal number in the range [0,4294967295]. Specifying the value 0 will disable hash randomization.
#
# https://docs.python.org/3/using/cmdline.html#cmdoption-r
# Turn on hash randomization. This option only has an effect if the PYTHONHASHSEED
# environment variable is set to 0, since hash randomization is enabled by default.
#
# https://docs.python.org/3/reference/datamodel.html#object.__hash__
# hash() truncates the value returned from an object’s custom __hash__() method to the size of a Py_ssize_t


import os.path
import platform

doc = "great blue product";


# Same for all variants
if(False):
    print(hash(doc), "hash(doc)")
    print(doc.__hash__(), "doc.__hash__()")
    print(hash("great blue product"), "hash(\"great blue product\")")


# PYTHONHASHSEED is relevant for hash calculations of different runs,
# not for different calls in same run
if(True):
    print(platform.python_version(), "platform.python_version()")
    print(os.environ.get("PYTHONHASHSEED"), "PYTHONHASHSEED")
    print(hash(doc), "hash(doc)")
    print(hash(doc), "hash(doc)")
    print()

# python3 dev-hash.py                                                                                                                                             :(
# 3.6.9 platform.python_version()
# None PYTHONHASHSEED
# 9038486907457221470 hash(doc)
# 9038486907457221470 hash(doc)
# 
# python3 dev-hash.py
# 3.6.9 platform.python_version()
# None PYTHONHASHSEED
# 7850955574087741919 hash(doc)
# 7850955574087741919 hash(doc)

# PYTHONHASHSEED=0 python3 dev-hash.py
# 3.6.9 platform.python_version()
# 0 PYTHONHASHSEED
# -3270336275963998745 hash(doc)
# -3270336275963998745 hash(doc)
# 
# adrian@nb-wilke ..L4U-Drift-detector-comparison/word2vec (git)-[main] % PYTHONHASHSEED=0 python3 dev-hash.py
# 3.6.9 platform.python_version()
# 0 PYTHONHASHSEED
# -3270336275963998745 hash(doc)
# -3270336275963998745 hash(doc)

# PYTHONHASHSEED=0 python3 -R dev-hash.py
# 3.6.9 platform.python_version()
# 0 PYTHONHASHSEED
# -3270336275963998745 hash(doc)
# -3270336275963998745 hash(doc)
# 
# PYTHONHASHSEED=0 python3 -R dev-hash.py
# 3.6.9 platform.python_version()
# 0 PYTHONHASHSEED
# -3270336275963998745 hash(doc)
# -3270336275963998745 hash(doc)
