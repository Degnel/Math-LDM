# expression_generator.py

import random
import math
from constants import NUMBER_RANGE, OPERATORS, EXPONENTS

class ExpressionGenerator:
    def __init__(self, max_depth=2, max_exponent=2):
        self.max_depth = max_depth
        self.max_exponent = max_exponent

    def generate_number(self):
        """Génère un nombre aléatoire dans la plage spécifiée."""
        return round(random.uniform(-50, 50), 2)

    def generate_operator(self):
        """Sélectionne un opérateur aléatoire."""
        return random.choice(OPERATORS)

    def generate_expression(self, depth=0):
        """Génère une expression mathématique valide."""
        if depth >= self.max_depth:
            return self.generate_simple_expression()

        choice = random.random()

        if choice < 0.25:
            left = self.generate_number()
            operator = self.generate_operator()
            right = self.generate_number()
            return f"{left}{operator}{right}"

        if choice < 0.5:
            base = self.generate_number()
            exp = random.choice(EXPONENTS[:self.max_exponent])
            return f"({base}**{exp})"

        if choice < 0.75:
            left = self.generate_expression(depth + 1)
            right = self.generate_expression(depth + 1)
            operator = self.generate_operator()
            return f"({left}{operator}{right})"

        left = self.generate_number()
        operator = random.choice(['*', '/'])
        right = self.generate_number()
        return f"({left}{operator}{right})"

    def generate_simple_expression(self):
        """Génère une simple expression sans parenthèses imbriquées."""
        left = self.generate_number()
        operator = self.generate_operator()
        right = self.generate_number()
        return f"{left}{operator}{right}"

    def calculate_value(self, expression: str) -> float:
        """Évalue l'expression et retourne le résultat numérique."""
        try:
            # Remplacer les espaces pour garantir la validité de l'expression
            expression = expression.replace(" ", "")
            return eval(expression)
        except Exception as e:
            # Gérer les erreurs de calcul (par exemple, division par zéro)
            print(f"Erreur lors du calcul de l'expression : {expression}")
            return None

    def generate_batch(self, batch_size: int):
        """Génère un batch de paires (expression, valeur)."""
        X = []
        y = []

        for _ in range(batch_size):
            expression = self.generate_expression()
            value = self.calculate_value(expression)

            # Ajouter l'expression et sa valeur au batch
            if value is not None:
                X.append(expression)
                y.append(value)

        return X, y