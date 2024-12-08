import random
import torch
import os


class MathExpressionDataset(torch.utils.data.Dataset):
    def __init__(
        self, num_samples=10000, file_path="./data/dataset.txt", force_recreate=False
    ):
        self.num_samples = num_samples
        self.file_path = file_path

        if (
            force_recreate
            or not os.path.exists(file_path)
            or self._file_size() < num_samples
        ):
            self._generate_and_save_data(force_recreate)

        with open(file_path, "r") as f:
            self.data = [line.strip() for line in f]

    def _file_size(self):
        """Retourne le nombre d'expressions dans le fichier."""
        if not os.path.exists(self.file_path):
            return 0
        with open(self.file_path, "r") as f:
            return sum(1 for _ in f)

    def _generate_and_save_data(self, force_recreate):
        """Génère et sauvegarde les données dans un fichier."""
        num_to_generate = (
            self.num_samples if force_recreate else self.num_samples - self._file_size()
        )
        if not os.path.exists(self.file_path):
            mode = "w"
        else:
            mode = "w" if force_recreate else "a"

        with open(self.file_path, mode) as f:
            for _ in range(num_to_generate):
                expression = self.generate_expression()
                f.write(expression + "\n")

    def generate_expression(self):
        """Génère une expression mathématique avec son résultat."""
        num1 = round(random.uniform(1, 10), 2)
        num2 = round(random.uniform(1, 10), 2)
        operator = random.choice(["+", "-", "*", "/"])
        try:
            result = eval(expression := f"{num1}{operator}{num2}")
        except ZeroDivisionError:
            result = float("inf")
        return f"{expression}={result:.2f}"
