import uuid

from django.db import models


class Conversation(models.Model):
    """A single ongoing chat thread. Identified by an opaque UUID that the
    browser stores in localStorage so a reload can resume the same thread.
    No auth for the prototype."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return f'Conversation {self.id}'


class Message(models.Model):
    ROLE_CHOICES = [
        ('user', 'user'),
        ('assistant', 'assistant'),
    ]

    conversation = models.ForeignKey(
        Conversation, related_name='messages', on_delete=models.CASCADE
    )
    role = models.CharField(max_length=16, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f'{self.role}: {self.content[:40]}'
