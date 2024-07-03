from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import sys

magn = np.arange(-6*6, 6*6+1, 2).reshape(-1,1)
# temp = float(sys.argv[1])
s_all = []
t_all = []
for temp in np.arange(20, 0.89, -0.1):
    if temp == 20.:
        data_t_size = int(20/temp*100000)
    else:
        data_t_size = int(20/temp*1000)
    print(temp, "S%.2f.txt"%temp, data_t_size)
    s = np.loadtxt("S%.2f.txt"%temp).reshape(-1,6*6)[:data_t_size].astype(np.float32)
    t = np.ones(data_t_size, dtype=np.float32)*temp
    data_t_size = min(data_t_size, len(s), len(t))
    s_all.append(s[:data_t_size])
    t_all.append(t[:data_t_size])

    # data = np.sum(s, axis=-1)
    # counter = Counter(data)
    # histgram = np.array([counter[m] for m in magn[:,0]])
    # P = histgram/np.sum(histgram)
    # free_energies = -np.log(P)*temp
    # plt.figure()
    # plt.plot(magn[:,0], free_energies)
    # with open(f"FES%.2f.dat"%temp, "w") as fp:
    #     for i in range(len(magn)):
    #         fp.write("%.2f  %.4f\n"%(magn[i,0], free_energies[i]))
    # plt.savefig("F-S%.2f.png"%temp)
    # plt.figure()
    # plt.plot(magn[:,0], histgram)
    # plt.savefig("H-S%.2f.png"%temp)
s_all = np.concatenate(s_all, axis=0)
t_all = np.concatenate(t_all, axis=0)
random_idx = np.arange(len(t_all))
# np.random.shuffle(random_idx)
print(s_all.shape, t_all.shape)
np.save("buffer_enhancelowTmaxT_ordered_dt0.1.npy", s_all[random_idx])
np.save("t_enhancelowTmaxT_ordered_dt0.1.npy", t_all[random_idx])

#s = np.load("buffer.npy")[:10000000]
#np.save("buffer-1000w.npy", s)
#s = np.load("t.npy")[:10000000]
#np.save("t-1000w.npy", s)


