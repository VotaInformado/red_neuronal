from django.db import models


class VoteChoices(models.TextChoices):
    # Ongoing status
    ABSENT = "ABSENT", "Ausente"
    ABSTENTION = "ABSTENTION", "Abstención"
    NEGATIVE = "NEGATIVE", "Negativo"
    POSITIVE = "POSITIVE", "Afirmativo"
    PRESIDENT = ("PRESIDENT", "Presidente")
