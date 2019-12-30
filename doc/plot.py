
#import numpy as np
import matplotlib.pyplot as plt

m1200 = [12.5, 14.2, 19.2]
i77 = [21.1, 22.5, 34.4]
gtx = [11.8, 13.2, 17.6]
i78 = [30.0, 33.4, 47.0]
x = [0, 1, 2]

plt.plot(x, m1200, marker="o", label="M1200")
plt.plot(x, gtx, marker="o", label="GTX 1050")
plt.plot(x, i77, marker="o", label="i7-7820HQ")
plt.plot(x, i78, marker="o", label="i7-8750H")
plt.legend()
plt.xticks([0,1,2], ["NY", "BAY", "COL"])
plt.ylabel("seconds")
plt.axis([0,2,0,50])

plt.show()
