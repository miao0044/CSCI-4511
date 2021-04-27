import numpy as np
import matplotlib.pyplot as plt


def calc_dist(cities):
    N = len(cities)
    dist = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            dist[i, j] = np.linalg.norm(cities[i,] - cities[j,])
    for i in range(N):
        for j in range(i):
            dist[i, j] = dist[j, i]
    
    return dist

def generate_new_sol(sol, method='reverse'):
    N = len(sol)
    new_sol = np.copy(sol)

    idx1, idx2 = np.random.choice(N, 2, replace=False)
    if method == 'swap':
        tmp = new_sol[idx1]
        new_sol[idx1] = new_sol[idx2]
        new_sol[idx2] = tmp
    elif method == 'reverse':
        idx1, idx2 = min([idx1, idx2]), max([idx1, idx2])
        for i in range(int((idx2-idx1)/2)):
            tmp = new_sol[idx2-i]
            new_sol[idx2-i] = new_sol[idx1+i]
            new_sol[idx1+i] = tmp
    elif method == 'insert':
        idx1, idx2 = min([idx1, idx2]), max([idx1, idx2])
        tmp = new_sol[idx2]
        for i in range(idx2, idx1, -1):
            new_sol[i] = new_sol[i-1]
        new_sol[idx1] = tmp

    return new_sol

def calcE(sol, dist):
    E = dist[sol[-1], sol[0]]
    for i in range(len(sol)-1):
        E += dist[sol[i], sol[i+1]]
    
    return E

def ga_algo(dist, max_epochs=200, pngname=''):
    def _crossover(s1, s2, ratio=.5):
        N = len(s1)
        l = int(ratio * N)
        s = [0 for _ in range(N)]
        start_pos = np.random.randint(N)
        s[start_pos : start_pos + l] = s1[start_pos : start_pos + l]
        c = [c for c in s2 if c not in s1[start_pos : start_pos + l]]
        s[: start_pos] = c[: start_pos]
        s[start_pos + l:] = c[start_pos : ]

        return np.array(s)

    def _mutate(s):
        return generate_new_sol(s, method='swap')

    def _select(sgroup):
        sgroup = [(s, calcE(s, dist)) for s in sgroup]
        sgroup = sorted(sgroup, key=lambda x:x[1])
        return sgroup[0], sgroup[1]

    Elist = []
    N = dist.shape[0]
    size = 30 * N
    s = np.random.permutation(N)
    solBest = s
    Emin = calcE(s, dist)
    for _ in range(max_epochs):
        Elist.append(Emin)
        sgroup = [_mutate(s) for _ in range(size)]
        (s1, E1), (s2, E2) = _select(sgroup)
        s = _crossover(s1, s2)
        E = calcE(s, dist)
        s_current, E_current = sorted([(s1, E1), (s2, E2), (s, E)], key=lambda x: x[1])[0]
        if E_current < Emin:
            Emin = E_current
            solBest = s_current

    plt.figure()
    plt.plot(Elist)
    plt.savefig('%s_ga.png' % pngname)
    return solBest, Emin

def sa_algo(dist, decay=.95, max_epochs=1000, pngname=''):
    Tinit = dist.max()
    Tstop = dist[dist != 0].min()

    N = dist.shape[0]
    sol = np.random.permutation(N)
    E = calcE(sol, dist)
    
    Emin = E
    Elast = E
    solBest = sol
    solLast = sol
    T = Tinit
    Elist = []
    while T > Tstop:
        Elist.append(Emin)
        for epoch in range(max_epochs):
            tmp_sol = generate_new_sol(solLast)
            tmp_E = calcE(tmp_sol, dist)
            delta = tmp_E - Elast
            if delta < 0:
                Elast = tmp_E
                solLast = tmp_sol
                if tmp_E < Emin:
                    Emin = tmp_E
                    solBest = tmp_sol
            else:
                prob = np.exp(-delta / T)
                if prob > np.random.rand():
                    Elast = tmp_E
                    solLast = tmp_sol
        T *= decay

    plt.figure()
    plt.plot(Elist)
    plt.savefig('%s_sa.png' % pngname)
    return solBest, Emin

if __name__ == '__main__':
    import os, sys
    x = []
    with open(sys.argv[1]) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                break
            x.append([])
            for item in line.split():
                x[-1].append(float(item))

    dist = np.array(x)
    
    sol, E = sa_algo(dist, pngname=sys.argv[1].rsplit('.', 1)[0])
    print('SA found solution {}'.format(sol), end='')
    print(', with a total distance = %.5f' % E)

    sol, E = ga_algo(dist, pngname=sys.argv[1].rsplit('.', 1)[0])
    print('GA found solution {}'.format(sol), end='')
    print(', with a total distance = %.5f' % E)


    gt_sol_file = sys.argv[1].rsplit('.', 1)[0] + '_sol.txt'
    if not os.path.exists(gt_sol_file):
        exit()
    with open(gt_sol_file) as f:
        sol = [int(t)-1 for t in f.readline().strip().split(',')]
        sol = np.array(sol)
        E = calcE(sol, dist)
        print('groundtruth solution has a total distance = %.5f' % E)