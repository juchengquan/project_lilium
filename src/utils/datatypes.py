class ConfigDict(dict):
    def update_values(self, dt: dict):
        _s_self = self.keys()
        _s_others =  dt.keys()
        _s_diff = set(_s_others).difference(set(_s_self))
        if not _s_diff:
            self.update(**dt)
            
        else:
            raise ValueError(f"Not supported keys in config: {list(_s_diff)}")
