import math


def evaluate_expression(expression):
    """
    Évalue une expression mathématique donnée.
    :param expression: Chaîne de caractères représentant une expression mathématique.
    :return: Résultat de l'évaluation.
    """
    try:
        # Restreindre les fonctions et les opérateurs autorisés
        allowed_locals = {
            "ln": math.log,
            "abs": abs,
            "pow": math.pow,
            "sqrt": math.sqrt,
            "exp": math.exp,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log10,
        }
        # Utiliser eval avec un environnement sécurisé
        return eval(expression, {"__builtins__": None}, allowed_locals)
    except Exception as e:
        raise ValueError(f"Invalid expression: {expression}. Error: {str(e)}")


def validate_expression(expression):
    """
    Valide une expression mathématique pour s'assurer qu'elle est correcte.
    :param expression: Chaîne de caractères représentant une expression mathématique.
    :return: Booléen indiquant si l'expression est valide.
    """
    try:
        evaluate_expression(expression)
        return True
    except ValueError:
        return False
