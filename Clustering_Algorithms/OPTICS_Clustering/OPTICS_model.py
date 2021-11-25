import numpy as np

class OPTICS:
    def __init__(self,min_samples=5, max_eps=np.inf,p=2,xi=0.05, predecessor_correction=True):
        self.max_eps = max_eps
        self.min_samples = min_samples
        self.p = p
        self.xi = xi
        self.predecessor_correction = predecessor_correction

    def Minkowski_Distance(self,x1,x2,p):
        return np.sum(np.abs(x1 - x2)**p,axis=-1)**(1/p)

    def consecutiveDiv(self,arr):
        with np.errstate(invalid="ignore"):
            return arr[:-1] / arr[1:]

    def xi_method(self):

        reachability_plot = np.hstack((self.reachability_[self.ordering_], np.inf))
        predecessor_plot = self.predecessor_[self.ordering_]

        xi_complement = 1 - self.xi
        sdas = []
        clusters = []
        index = 0
        mib = 0

        ratio = self.consecutiveDiv(reachability_plot)
        downward = ratio > 1
        upward = ratio < 1
        steep_upward = ratio <= xi_complement
        steep_downward = ratio >= 1 / xi_complement
        
        for steep_index in iter(np.flatnonzero(steep_upward | steep_downward)):
            if steep_index < index:
                continue

            mib = max(mib, np.max(reachability_plot[index : steep_index + 1]))
            if np.isinf(mib):
                sdas = []
            else:
                sdas = [sda for sda in sdas if mib <= reachability_plot[sda["start"]] * xi_complement]
                for sda in sdas:
                    sda["mib"] = max(sda["mib"], mib)

            if steep_downward[steep_index]:
                D_start = steep_index
                D_end = steep_index
                non_xward_points = 0
                for i in range(steep_index,len(steep_downward)):
                    if steep_downward[i]:
                        non_xward_points = 0
                        D_end = i
                    elif not upward[i]:
                        non_xward_points += 1
                        if non_xward_points > self.min_samples:
                            break
                    else:
                        break
                sdas.append({"start": D_start, "end": D_end, "mib": 0.0})
                index = D_end + 1
                mib = reachability_plot[index]

            else:
                U_end = steep_index
                non_xward_points = 0
                for i in range(steep_index,len(steep_upward)):
                    if steep_upward[i]:
                        non_xward_points = 0
                        U_end = i
                    elif not downward[i]:
                        non_xward_points += 1
                        if non_xward_points > self.min_samples:
                            break
                    else:
                        break
                index = U_end + 1
                mib = reachability_plot[index]

                U_clusters = []
                for sda in sdas:
                    c_start = sda["start"]
                    c_end = U_end

                    if reachability_plot[c_end + 1] * xi_complement < sda["mib"]:
                        continue

                    if reachability_plot[sda["start"]] * xi_complement >= reachability_plot[c_end + 1]:
                        while (reachability_plot[c_start + 1] > reachability_plot[c_end + 1] and c_start < sda["end"]):
                            c_start += 1
                    
                    elif reachability_plot[c_end + 1] * xi_complement >= reachability_plot[sda["start"]]:
                        while reachability_plot[c_end - 1] > reachability_plot[sda["start"]] and c_end > steep_index:
                            c_end -= 1

                    if self.predecessor_correction:
                        con = True
                        while c_start < c_end:
                            if (reachability_plot[c_start] > reachability_plot[c_end]) or (self.ordering_[predecessor_plot[c_end]] in self.ordering_[c_start: c_end]):
                                con = False
                                break
                            c_end -= 1
                    if con or (c_end - c_start + 1 < self.min_samples) or (c_start > sda["end"]) or (c_end < steep_index):
                        continue

                    U_clusters.append((c_start, c_end))

                U_clusters.reverse()
                clusters += U_clusters

        labels = np.full(len(self.ordering_), -1, dtype=int)
        label = 0
        for c in clusters:
            if not np.any(labels[c[0] : (c[1] + 1)] != -1):
                labels[c[0] : (c[1] + 1)] = label
                label += 1
        labels[self.ordering_] = labels.copy()

        return labels,np.array(clusters)

    def fit(self, X): 
        core_distances_ = np.sort(self.Minkowski_Distance(X,np.expand_dims(X, axis=1),self.p),axis=-1)[:,self.min_samples - 1]
        core_distances_[core_distances_ > self.max_eps] = np.inf
        self.core_distances_ = np.around(core_distances_,decimals=np.finfo(core_distances_.dtype).precision)
    
        self.reachability_ = np.full(X.shape[0], np.inf)
        self.predecessor_ = np.full(X.shape[0], -1)
        processed = np.zeros(X.shape[0], dtype=bool)
        self.ordering_ = np.zeros(X.shape[0], dtype=int)
        for ordering_idx in range(X.shape[0]):
            index = np.where(processed == 0)[0]
            point_index = index[np.argmin(self.reachability_[index])]

            processed[point_index] = True
            self.ordering_[ordering_idx] = point_index
            if self.core_distances_[point_index] != np.inf:
                indices = np.where(self.Minkowski_Distance(X[point_index],X,self.p) <= self.max_eps)[0]
                unproc = indices[~processed[indices]]
                if unproc.size:
                    dists = self.Minkowski_Distance(X[point_index],X[unproc],self.p)
                    rdists = np.maximum(dists, self.core_distances_[point_index])
                    rdists = np.around(rdists, decimals=np.finfo(rdists.dtype).precision)
                    improved = np.where(rdists < self.reachability_[unproc])
                    self.reachability_[unproc[improved]] = rdists[improved]
                    self.predecessor_[unproc[improved]] = point_index

        self.labels_,self.cluster_hierarchy_ = self.xi_method()
        

