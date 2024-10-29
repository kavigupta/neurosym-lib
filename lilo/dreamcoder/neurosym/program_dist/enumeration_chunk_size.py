# Let c be the chunk size, h be the likelihood of the program we are trying to
# find, and T be the number of iterations we need to do.
#
# We have that the total time taken is
#     sum_{t=0}^{T - 1} e^{c * t} + f * e^{c * T}
# where f is the fraction of the last chunk. We can compute this in expectation
# as 1/2, which leads us to
#     sum_{t=0}^{T - 1} e^{c * t} + e^{c * T} / 2
# which can be computed as
#     (e^{c * (T + 1)} - 1) / (e^c - 1) - e^{c * T} / 2
# which can be approximated closely as
#     e^{cT} * e^c / (e^c - 1) - e^{c * T} / 2
# which is equal to
#     e^{cT} * (e^c / (e^c - 1) - 1/2)
# we then know that T = ceil(h / c), and letting g = T - h / c, we have
#     e^{cT} = e^{c * (h / c + g)} = e^h * e^{c * g}
# if we assume g is uniformly distributed between 0 and 1, we have
#     e^{c * g} = (e^c - 1) / c
# we then have that the expected time is
#     (e^c - 1) / c * (e^c / (e^c - 1) - 1/2)
# which is optimized per wolfram at
#     W(1/e) + 1
# which is approximately 1.28.
DEFAULT_CHUNK_SIZE = 1.278
