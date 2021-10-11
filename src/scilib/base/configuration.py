class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def rec_update(self, other: 'Config') -> 'Config':
        for k in other:
            if k in self:
                if isinstance(self[k], Config):
                    self[k] = self[k].rec_update(other[k])
                else:
                    self[k] = other[k]
            else:
                self[k] = other[k]

        return self
