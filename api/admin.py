from django.contrib import admin

from .models import Conversation, Message


class MessageInline(admin.TabularInline):
    model = Message
    extra = 0
    readonly_fields = ('role', 'content', 'created_at')
    can_delete = True


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'message_count', 'updated_at', 'created_at')
    readonly_fields = ('id', 'created_at', 'updated_at')
    ordering = ('-updated_at',)
    inlines = [MessageInline]

    @admin.display(description='First message')
    def title(self, obj):
        first = obj.messages.filter(role='user').first()
        if not first:
            return '(empty)'
        return first.content[:60] + ('…' if len(first.content) > 60 else '')

    @admin.display(description='Messages')
    def message_count(self, obj):
        return obj.messages.count()


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('conversation', 'role', 'short_content', 'created_at')
    list_filter = ('role', 'created_at')
    search_fields = ('content',)
    readonly_fields = ('created_at',)

    @admin.display(description='Content')
    def short_content(self, obj):
        return obj.content[:80] + ('…' if len(obj.content) > 80 else '')
