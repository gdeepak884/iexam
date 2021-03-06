from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class Student(models.Model):

    user = models.OneToOneField(User,on_delete=models.CASCADE)

    Institution = models.CharField(max_length=256)
    State = models.CharField(max_length=256)
    Country = models.CharField(max_length=256)

    def __str__(self):
        return self.user.username
