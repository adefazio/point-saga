
import optimization.run
from optimization.run import runit
import pstats, cProfile

import pyximport
pyximport.install()

cProfile.runctx("runit()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
