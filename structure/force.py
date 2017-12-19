    def force_i(self,i,distances,vecs,n_list):
        # pref_sep = [self.age[i]*(L0-2*EPS) +2*EPS if self.mother[i] == self.mother[j] and self.age[i] <1.0
        #                 else L0 for j in n_list]
        # import ipdb; ipdb.set_trace()
        pref_sep = RHO+0.5*ALPHA*(self.age[n_list]+self.age[i])
        pref_sep[pref_sep>2.] = 2.
        return (MU*vecs*np.repeat((distances-pref_sep)[:,np.newaxis],2,axis=1)).sum(axis=0)

    def dr(self,dt):
        distances,vecs,neighbours = self.mesh.distances,self.mesh.unit_vecs,self.mesh.neighbours
        return (dt/ETA)*np.array([self.force_i(i,dist,vec,neigh) for i,(dist,vec,neigh) in enumerate(zip(distances,vecs,neighbours))])
