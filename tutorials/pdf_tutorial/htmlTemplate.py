css = '''
<style>
.chat-message {
    padding: 10px;
    margin: 10px;
    border-radius: 10px;
    display: inline-block;
    max-width: 70%;
}
.chat-message.user {
    background-color: #f4f4f4;
}
.chat-message.bot {
    background-color: #f4f4f4;
}
.chat-message .avatar {
    width: 50px;
}
.chat-message .avatar img { 
    width: 100%;
    border-radius: 50%;
}
.chat-message .message {
    margin: 0;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://image.flaticon.com/icons/png/512/1946/1946433.png" />
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''	

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://image.flaticon.com/icons/png/512/1946/1946433.png" />
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''