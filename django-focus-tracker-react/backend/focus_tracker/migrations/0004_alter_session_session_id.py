# Generated by Django 4.2.5 on 2024-03-16 20:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('focus_tracker', '0003_session'),
    ]

    operations = [
        migrations.AlterField(
            model_name='session',
            name='session_id',
            field=models.CharField(max_length=255, null=True),
        ),
    ]