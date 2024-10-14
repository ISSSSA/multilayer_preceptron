class ClassLabelMapping:
    MAPPING = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4,
        'F': 5,
        'G': 6,
        'H': 7,
        'I': 8,
        'J': 9,
    }

    @staticmethod
    def from_input_data(input_data):
        return ClassLabelMapping.MAPPING[input_data.label]

    @staticmethod
    def load_labels_from_file(file_path):
        label_mapping = {}
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    # Предполагается, что файл содержит метки в формате: "метка,число"
                    label, number = line.strip().split(',')
                    label_mapping[int(number)] = label
        except Exception as e:
            raise RuntimeError(f"Error loading labels from file: {e}")

        return label_mapping
