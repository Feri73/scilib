class Config(dict):
    def __getattr__(self, item):
        if item in self:
            return self[item]
        return super(Config, self).__getattr__(item)
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
