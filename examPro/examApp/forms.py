from django import forms
from django.contrib.auth.models import User
from examApp.models import Student

class UserForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput())

    class Meta():
        model = User
        fields = ['username','email','password']

class StudentForm(forms.ModelForm):


    class Meta():
        model = Student
        fields = ['Institution','State','Country']
