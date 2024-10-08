class InputData:
    def __init__(self, data, label):
        self._data = data  # используем _data для обозначения приватного атрибута
        self.label = label

    @property
    def data(self):
        return self._data  # используем свойство для доступа к данным
