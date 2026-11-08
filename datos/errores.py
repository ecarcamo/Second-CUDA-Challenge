"""error comun para todo lo que no cuadra con los datos o el config"""


class ErrorDatos(Exception):
    """falla esperada: el script la reporta y termina con codigo 1"""
